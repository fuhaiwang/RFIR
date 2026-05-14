#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import csv
import math
import torch
import numpy as np
import torch.nn.functional as F
from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera, Antenna
from utils.sh_utils import eval_sh
from utils.loss_utils import ssim, bilateral_smooth_loss, second_order_edge_aware_loss, tv_loss, first_order_edge_aware_loss, first_order_loss, first_order_edge_aware_norm_loss
from utils.image_utils import psnr
from utils.graphics_utils import fibonacci_sphere_sampling, rgb_to_srgb, srgb_to_rgb
# from .r3dg_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # 重新生成 工具库！！
from .diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# import scipy.constants as sc


def to_list(x):
    if torch.is_tensor(x):
        return x.detach().cpu().view(-1).tolist()
    elif isinstance(x, np.ndarray):
        return x.reshape(-1).tolist()
    elif isinstance(x, (int, float, np.number)):
        return [x]
    else:
        return list(x)


def render_view(viewpoint_antenna, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                is_training, dict_params, d_reflection_coe, d_reflection_phase, d_roughness, frequency, iteration, 
                use_trained_exp=False, scaling_modifier=1.0, override_color=None, separate_sh=False):
                
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_antenna.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_antenna.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_antenna.image_height),
        image_width=int(viewpoint_antenna.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,  # ??
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_antenna.world_view_transform,  # 平移矩阵
        projmatrix=viewpoint_antenna.full_proj_transform,  # 投影矩阵
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_antenna.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity_ = pc.get_opacity


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation 

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    if override_color is None:
        if pipe.compute_SHs_python:
            shs_view = pc.get_shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1).pow_(2))
            dir_pp_normalized = F.normalize(viewpoint_antenna.camera_center.repeat(means3D.shape[0], 1) - means3D,
                                            dim=-1)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_shs_dc, pc.get_shs_rest
            else:
                shs = pc.get_shs

    #V 47
    opacity = pc.get_opacity
    alpha = 0.5
    reflection_coe = pc.get_reflection_coe * torch.exp(alpha * d_reflection_coe)
    # reflection_coe = pc.get_reflection_coe * (1 + d_reflection_coe)   # [0,10]
    reflection_phase = pc.get_reflection_phase + d_reflection_phase # [0,2pi]
    roughness = pc.get_roughness * torch.exp(0.7 * d_roughness) 

    roughness = 1.0 + F.softplus(roughness - 1.0)


    normal = pc.get_normal  # 法向量
    viewdirs = F.normalize(means3D - viewpoint_antenna.camera_center.repeat(pc.get_shs.shape[0], 1), dim=-1)   # v1.0-4
    scattered_dists = torch.norm(means3D - viewpoint_antenna.camera_center.repeat(pc.get_shs.shape[0], 1), dim=-1, keepdim=True)

    S_Tx = viewpoint_antenna.Tx_power

    incident_dists = pc._incident_distances

    frequencies = frequency.clone().detach().to(dtype=torch.float32, device="cuda")

    H_amp, Hi_phases = compute_channel_H(incident_dists, scattered_dists, frequencies)

    cov_all = pc.get_covariance()
    incov_all = pc.get_inverse_covariance()

    if is_training:
        brdf_RF = rendering_equation_RF_txGSrx(viewdirs, cov_all, incov_all,
                 S_Tx, H_amp, Hi_phases, roughness, normal.detach(), reflection_coe, reflection_phase,
                 visibility_precompute=pc._visibility_tracing, incident_distances=pc._incident_distances,
                 incident_dirs=pc._incident_dirs)
    else:
        brdf_RF = rendering_equation_RF_txGSrx(viewdirs, cov_all, incov_all,
                 S_Tx, H_amp, Hi_phases, roughness, normal.detach(), reflection_coe, reflection_phase,
                 visibility_precompute=pc._visibility_tracing, incident_distances=pc._incident_distances,
                 incident_dirs=pc._incident_dirs)

    d1, d2, d3 = brdf_RF.shape


    features = torch.cat([pc.get_roughness,
                          pc.get_reflection_coe,
                          pc.get_reflection_phase,
                          ], dim=-1)
    all_results = {}
    rendered_CSI_allfreq = torch.empty((d2), dtype=torch.cfloat, device=brdf_RF.device)
    
    for i in range(0, d2):
        # print(i)
        if separate_sh:
            rendered_RF_raw, rendered_feature, radii, depth_image = rasterizer(
                means3D=means3D,
                means2D=means2D,
                dc=dc,
                shs=shs,
                colors_precomp=brdf_RF[:, i, :],
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
                features=features)
        else:
            rendered_RF_raw, rendered_feature, radii, depth_image = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=None,
                colors_precomp=brdf_RF[:, i, :],
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
                features=features)

        subarrary_amp_pha, raw, col = rendered_RF_raw.shape
        rf_split = rendered_RF_raw.view(2, subarrary_amp_pha//2, raw, col)  
        rf_complex = rf_split[0].float() * (torch.cos(rf_split[1]) + 1j * torch.sin(rf_split[1]))
        rendered_CSI_allfreq[i] = rf_complex.sum(dim=(1, 2))

    all_results = {"render": rendered_CSI_allfreq}

    return all_results


def calculate_loss(viewpoint_antenna, pc, all_results, opt, batch_size, iteration, freq_idx, direct_light_env_light, p_scale):

    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    loss_all = 0
    gt_raw_all = viewpoint_antenna.original_image.cuda()
    rendered_CSIs = all_results["render"]

    gt_raw = gt_raw_all[(batch_size*freq_idx):(batch_size*(freq_idx+1))]

    lambda_abs = 1

    for i in range(0, len(rendered_CSIs)):
        rendered_CSI = rendered_CSIs[i]

        gt_CSI = (gt_raw[i, 0] + 1j * gt_raw[i, 1] ) * p_scale
        sampled_gt_CSI = gt_CSI.reshape(-1, 1).to("cuda")


        rendered_abs = torch.sqrt(rendered_CSI.real ** 2 + rendered_CSI.imag ** 2 + 1e-12)
        gt_abs = torch.sqrt(sampled_gt_CSI.real ** 2 + sampled_gt_CSI.imag ** 2 + 1e-12)
        L_abs = torch.abs(rendered_abs - gt_abs) # l2
        tb_dict["l_abs"] = L_abs.item()


        loss = L_abs * lambda_abs
        loss_all += loss

    tb_dict["loss"] = loss_all.item()

    return loss_all, tb_dict



def sig2mse_complex(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    diff_real = torch.real(x) - torch.real(y)
    diff_imag = torch.imag(x) - torch.imag(y)
    return torch.mean(diff_real ** 2 + diff_imag ** 2)


def rendering_equation_RF_txGSrx_v1(viewdirs, cov, incov, direct_signal, Hi_phases, roughness, normals, reflection_coe, reflection_phase,
                                 visibility_precompute=None, incident_distances=None, incident_dirs=None):
    eps = 1e-20
    # incident_dir_perGS = incident_dirs
    incident_areas = ellipsoid_projection_area(incident_dirs, cov, incov)

    # compute spherical area denominator
    abs_frac = 4.0 * math.pi * (incident_distances.clamp(min=eps).pow_(2))  # [M, N]
    incident_signals_pow = incident_areas.unsqueeze(-1) / abs_frac * visibility_precompute * direct_signal

    f_s, _ = Degli_Esposti_scattering(normals, viewdirs, incident_dirs, roughness, BSDF_class='Degli-Esposti')
    
    pbr = torch.empty(f_s.shape + (2,), device=f_s.device, dtype=f_s.dtype)

    pbr[..., 0] = f_s * torch.sqrt(incident_signals_pow + eps) * reflection_coe
    pbr[..., 1] = reflection_phase + Hi_phases.squeeze(1)

    return pbr 


def rendering_equation_RF_txGSrx(viewdirs, cov, incov, direct_signal, H_amp, H_phases, roughness, normals, reflection_coe, reflection_phase, visibility_precompute=None, incident_distances=None, incident_dirs=None):
    eps = 1e-20

    incident_dir_perGS = incident_dirs
    incident_areas = ellipsoid_projection_area(incident_dir_perGS, cov, incov)  
    abs_frac = 4.0 * math.pi * (incident_distances.clamp(min=eps).pow_(2))  
    incident_signals_pow = incident_areas.unsqueeze(-1) / abs_frac * visibility_precompute * direct_signal

    f_s, _ = Degli_Esposti_scattering(normals, viewdirs, incident_dir_perGS, roughness, BSDF_class='Degli-Esposti')
    
    pbr = torch.empty(f_s.shape + (2,), device=f_s.device, dtype=f_s.dtype)

    pbr[..., 0] = f_s * torch.sqrt(incident_signals_pow + eps) * reflection_coe * H_amp.view(-1, 1)
    
    pbr[..., 1] = reflection_phase + H_phases.squeeze(1)

    return pbr



def ellipsoid_projection_area(incident_dir_perGS, cov, incov, eps=1e-12):

    dirs = incident_dir_perGS / (incident_dir_perGS.norm(dim=-1, keepdim=True) + eps)

    cov_matrix = torch.zeros(cov.shape[0], 3, 3, device=cov.device, dtype=cov.dtype)
    cov_matrix = cov_matrix.triu(diagonal=0)
    cov_matrix[..., 0, 0] = cov[..., 0]
    cov_matrix[..., 0, 1] = cov[..., 1]
    cov_matrix[..., 0, 2] = cov[..., 2]
    cov_matrix[..., 1, 1] = cov[..., 3]
    cov_matrix[..., 1, 2] = cov[..., 4]
    cov_matrix[..., 2, 2] = cov[..., 5]

    batch_size = incov.shape[0]
    incov_matrix = torch.zeros(batch_size, 3, 3, device=incov.device, dtype=incov.dtype)
    incov_matrix = incov_matrix.triu(diagonal=0)

    incov_matrix[..., 0, 0] = incov[..., 0]
    incov_matrix[..., 0, 1] = incov[..., 1]
    incov_matrix[..., 0, 2] = incov[..., 2]
    incov_matrix[..., 1, 1] = incov[..., 3]
    incov_matrix[..., 1, 2] = incov[..., 4]
    incov_matrix[..., 2, 2] = incov[..., 5]

    sign, logabsdet = torch.linalg.slogdet(cov_matrix)
    if torch.any(sign <= 0):
        cov_matrix = cov_matrix + eps * torch.eye(3, device=cov_matrix.device, dtype=cov_matrix.dtype)
        sign, logabsdet = torch.linalg.slogdet(cov_matrix)

    n_S_n = torch.einsum('...i,...ij,...j->...', dirs, incov_matrix, dirs)
    n_S_n = torch.clamp(n_S_n, min=eps)

    log_area = 0.5 * (logabsdet - torch.log(n_S_n)) + torch.log(
        torch.tensor(torch.pi, device=cov_matrix.device, dtype=cov_matrix.dtype))
    area = torch.exp(log_area)

    return area


def Degli_Esposti_scattering_v1(normal, ws, wi, roughness, BSDF_class):

    if BSDF_class == "Degli-Esposti":
        WI = F.normalize(wi, dim=-1)
        N = F.normalize(normal, dim=-1)
        NoWI = torch.sum(N * WI, dim=-1, keepdim=True).clamp_(1e-6, 1)  # cos(theta_i)

        WS = F.normalize(ws, dim=-1)
        NoWS = torch.sum(WS * N, dim=-1, keepdim=True)
        N = N * NoWS.sign()
        NoWS = torch.sum(N * WS, dim=-1, keepdim=True).clamp_(1e-6, 1)

        R = WI - 2 * NoWI * N   #
        WSoR = torch.sum(WS * R, dim=-1, keepdim=True).clamp_(1e-6, 1)


        eps = 1e-6
        WSoR_safe = WSoR.clamp(-1.0 + eps, 1.0 - eps)
        base = (1.0 + WSoR_safe) * 0.5  # ∈ (0, 1)
        rough_safe = torch.clamp_min(roughness, 1e0)

        f_pow_out = torch.pow(base.clamp_min(eps), rough_safe)
        f_thetai_phii_thetas_phis_ = f_pow_out.clone()

        f_thetai_phii_thetas_phis_ *= NoWS.pow_(0.5)             # (14)

        k_alphaR = 1 / (0.07937 * roughness + 0.1745)
        F_alphaR_wi = NoWI.pow_(0.5) * k_alphaR   # (16)

        beta = NoWI / F_alphaR_wi * f_thetai_phii_thetas_phis_ 
        alpha = np.pi
    else:
        beta = 1/np.pi
        alpha = np.pi

    return torch.sqrt(beta), alpha

def Degli_Esposti_scattering(normal, ws, wi, roughness, BSDF_class):

    if BSDF_class == "Degli-Esposti":
        WI = F.normalize(wi, dim=-1)
        N = F.normalize(normal, dim=-1)
        NoWI = torch.sum(N * WI, dim=-1, keepdim=True).clamp_(1e-6, 1)

        WS = F.normalize(ws, dim=-1)   # 视线方向
        NoWS = torch.sum(WS * N, dim=-1, keepdim=True)
        N = N * NoWS.sign()
        NoWS = torch.sum(N * WS, dim=-1, keepdim=True).clamp_(1e-6, 1)

        R = WI - 2 * NoWI * N   #
        WSoR = torch.sum(WS * R, dim=-1, keepdim=True).clamp_(1e-6, 1)  

        eps = 1e-6
        WSoR_safe = WSoR.clamp(-1.0 + eps, 1.0 - eps)
        base = (1.0 + WSoR_safe) * 0.5  # ∈ (0, 1)
        rough_safe = torch.clamp_min(roughness, 1e0)

        f_pow_out = torch.pow(base.clamp_min(eps), rough_safe)  #
        f_thetai_phii_thetas_phis_ = f_pow_out.clone()

        f_thetai_phii_thetas_phis_ *= NoWS.pow_(0.5)             # (14)

        k_alphaR = 1 / (0.07937 * roughness + 0.1745)
        F_alphaR_wi = NoWI.pow_(0.5) * k_alphaR   # (16)

        beta = NoWI / F_alphaR_wi * f_thetai_phii_thetas_phis_ 
        alpha = np.pi  
    else:
        beta = 1/np.pi
        alpha = np.pi

    return  torch.sqrt(torch.clamp(beta, min=0.0) + 1e-20), alpha
    


def compute_channel_H_v1(distances, freqs):
    distances = torch.as_tensor(distances, dtype=torch.float32)
    freqs = torch.as_tensor(freqs, dtype=torch.float32)

    c = 299792458.0
    wavelength = c / freqs

    phase_shift_raw = 2 * torch.pi * distances[..., None] / wavelength[None, :]

    phase_shift = phase_shift_raw.squeeze(-1)

    amp_decay = 1 / distances[..., None]

    return amp_decay, phase_shift  # 返回幅度和相位


def compute_channel_H(distances_i, distances_o, freqs):
    distances = distances_i + distances_o
    distances_o = torch.as_tensor(distances_o, dtype=torch.float32)
    distances = torch.as_tensor(distances, dtype=torch.float32)
    freqs = torch.as_tensor(freqs, dtype=torch.float32)
    c = 299792458.0
    wavelength = c / freqs

    phase_shift_raw = 2 * torch.pi * distances[..., None] / wavelength[None, :]
    phase_shift = torch.fmod(phase_shift_raw, 2 * torch.pi)


    amp_decay = 1 / distances_o[..., None]

    return amp_decay, phase_shift


def render_RF_bb(viewpoint_antenna: Antenna, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                       d_reflection_coe=None, d_reflection_phase=None, d_roughness=None,  batch_size=None, frequency=None,
                       scaling_modifier=1.0, override_color=None, opt: OptimizationParams = False,
                       is_training=False, dict_params=None, iteration=None, **kwargs):

    all_results = render_view(viewpoint_antenna, pc, pipe, bg_color, is_training, dict_params, d_reflection_coe, d_reflection_phase, d_roughness, frequency, iteration,
                        use_trained_exp=False)

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_antenna, pc, all_results, opt, batch_size, iteration, freq_idx=dict_params['freq_idx'], direct_light_env_light=dict_params['env_light'], p_scale=dict_params['power_scale'])
        all_results["tb_dict"] = tb_dict
        all_results["loss"] = loss

    return all_results






















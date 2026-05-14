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
# from .r3dg_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from .diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# import scipy.constants as sc
from bvh import RayTracer


def render_view(viewpoint_antenna, pc: GaussianModel, pipe, bg_color: torch.Tensor, is_training=False, dict_params=None,
                use_trained_exp=False, scaling_modifier=1.0, override_color=None, separate_sh=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
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
        bg=bg_color,       # ??
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_antenna.world_view_transform,
        projmatrix=viewpoint_antenna.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_antenna.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
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
    colors_precomp = None
    if override_color is None:
        if pipe.compute_SHs_python:
            shs_view = pc.get_shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1).pow_(2))
            dir_pp_normalized = F.normalize(viewpoint_antenna.camera_center.repeat(means3D.shape[0], 1) - means3D,
                                            dim=-1)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_shs_dc, pc.get_shs_rest
            else:
                shs = pc.get_shs
    else:
        colors_precomp = override_color

    reflection_coe = pc.get_reflection_coe  # [0,10]
    aa = np.array(reflection_coe.detach().cpu())

    count_above_0 = np.sum(aa > 0)
    count_above_49 = np.sum(aa > 49)

    reflection_phase = pc.get_reflection_phase  # [0,2pi]
    bb = np.array(reflection_phase.detach().cpu())

    roughness = pc.get_roughness
    normal = pc.get_normal
    cc = np.array(roughness.detach().cpu())


    viewdirs = F.normalize(means3D - viewpoint_antenna.camera_center.repeat(pc.get_shs.shape[0], 1), dim=-1)   # v1.0-4
    scattered_dists = torch.norm(means3D - viewpoint_antenna.camera_center.repeat(pc.get_shs.shape[0], 1), dim=-1, keepdim=True)


    direct_signal = 1e3*100


    incident_dist = pc._incident_distances

    frequencies = torch.tensor([2.4e9], dtype=torch.float32, device="cuda")

    H_amp, H_phases = compute_channel_H(incident_dist+scattered_dists, frequencies)

    cov_all = pc.get_covariance()
    incov_all = pc.get_inverse_covariance()

    if is_training:
        brdf_RF, extra_results = rendering_equation_RF_txGSrx(viewdirs, cov_all, incov_all,
                 direct_signal, H_amp, H_phases, roughness, normal.detach(), reflection_coe, reflection_phase,
                 visibility_precompute=pc._visibility_tracing, incident_distances=pc._incident_distances,
                 incident_dirs=pc._incident_dirs)
    else:
        chunk_size = 100000
        brdf_RF = []
        extra_results = []
        for i in range(0, means3D.shape[0], chunk_size):
            _brdf_RF, _extra_results = rendering_equation_RF_txGSrx(viewdirs[i:i + chunk_size], cov_all[i:i + chunk_size], incov_all[i:i + chunk_size],
                 direct_signal, H_amp[i:i + chunk_size], H_phases[i:i + chunk_size], roughness[i:i + chunk_size], normal[i:i + chunk_size].detach(), reflection_coe[i:i + chunk_size], reflection_phase[i:i + chunk_size],
                 visibility_precompute=pc._visibility_tracing[i:i + chunk_size], incident_distances=pc._incident_distances[i:i + chunk_size],
                 incident_dirs=pc._incident_dirs[i:i + chunk_size])

            brdf_RF.append(_brdf_RF)
            extra_results.append(_extra_results)

        brdf_RF = torch.cat(brdf_RF, dim=0)

        extra_results = {k: torch.cat([_extra_results[k] for _extra_results in extra_results], dim=0) for k in extra_results[0]}
        torch.cuda.empty_cache()

    xyz_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)

    d1, d2, d3 = brdf_RF.shape
    brdf_RF_set = brdf_RF.reshape(d1, d2 * d3)  # N * 16  (2*8)

    if is_training:
        features = torch.cat([roughness,
                              reflection_coe,
                              reflection_phase,
                              ], dim=-1)
    else:
        features = torch.cat([roughness,  # 1,1,1
                              reflection_coe,
                              reflection_phase,
                              ], dim=-1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if separate_sh:
        rendered_RF_raw, rendered_feature, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            dc=dc,
            shs=shs,
            colors_precomp=brdf_RF_set,
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
            colors_precomp=brdf_RF_set,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            features=features)

    feature_dict = {}
    if is_training:
        rendered_roughness, rendered_reflection_coe, rendered_reflection_phase \
            = rendered_feature.split([1, 1, 1], dim=0)
        feature_dict.update({"roughness": rendered_roughness,
                             "reflection_coe": rendered_reflection_coe,
                             "reflection_phase": rendered_reflection_phase
                             })
    else:
        rendered_roughness, rendered_reflection_coe, rendered_reflection_phase \
            = rendered_feature.split([1, 1, 1], dim=0)
        feature_dict.update({"roughness": rendered_roughness,
                             "reflection_coe": rendered_reflection_coe,
                             "reflection_phase": rendered_reflection_phase
                             })

    subarrary_amp_pha, raw, col = rendered_RF_raw.shape
    rf_split = rendered_RF_raw.view(2, subarrary_amp_pha//2, raw, col)

    amp = rf_split[0].float()
    phase = rf_split[1]
    rf_complex = amp * (torch.cos(phase) + 1j * torch.sin(phase))

    rendered_CSI = rf_complex.sum(dim=(1, 2))
    rf_complex_abs = np.abs(rf_complex.cpu().detach().numpy())*1e0

    G_Tx = 1
    G_Tx = torch.tensor(G_Tx, dtype=torch.float32, device="cuda")
    c = 3e8

    LOS_dist = torch.norm(viewpoint_antenna.camera_center - viewpoint_antenna.Tx_pos, dim=-1, keepdim=True)


    wavelength = c / frequencies
    Ho_phases = torch.exp(-1j * 2 * torch.pi * LOS_dist / wavelength)

    Ho_amp = torch.sqrt(G_Tx) * (wavelength / (4 * torch.pi * LOS_dist))
    w_los = 2

    h_los = Ho_amp * w_los * Ho_phases


    raytracer = RayTracer(pc.get_xyz, pc.get_scaling, pc.get_rotation)
    gaussians_inverse_covariance = pc.get_inverse_covariance()
    gaussians_opacity = pc.get_opacity[:, 0]
    gaussians_normal = pc.get_normal
    incident_dirs = F.normalize(viewpoint_antenna.Tx_pos - viewpoint_antenna.Rx_pos, dim=-1)
    distance_los = LOS_dist # .to(torch.float32)

    trace_results = raytracer.trace_visibility_segment(
        viewpoint_antenna.Rx_pos.to(dtype=torch.float32),
        incident_dirs.to(dtype=torch.float32),
        means3D,
        gaussians_inverse_covariance,
        gaussians_opacity,
        gaussians_normal,
        distance_los)
    incident_visibility = trace_results["visibility"]
    rendered_CSI_LOS = h_los * incident_visibility
    rendered_CSI_ = rendered_CSI_LOS + rendered_CSI
    # print(rendered_CSI_LOS)

    # print(rendered_CSI)
    power_ratio = calculate_los_power_ratio(rendered_CSI_, rendered_CSI_LOS)

    results = {"render": rendered_CSI_,
               "viewspace_points": screenspace_points,
               "visibility_filter": radii > 0,
               "radii": radii
               }

    results.update(feature_dict)
    return results


def calculate_los_power_ratio(rendered_CSI_, rendered_CSI_LOS):

    los_power = torch.abs(rendered_CSI_LOS).pow(2).sum()

    total_power = torch.abs(rendered_CSI_).pow(2).sum()

    power_ratio = los_power / total_power
    
    return power_ratio.item()

    
def calculate_loss(viewpoint_antenna, pc, results, opt, direct_light_env_light):

    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    rendered_CSI = results["render"].unsqueeze(-1).to("cuda") * 1e4

    gt_raw = viewpoint_antenna.original_image.cuda()
    N = gt_raw.size(0)
    center_idx = N // 2

    gt_CSI = (gt_raw[center_idx, 0] + 1j * gt_raw[center_idx, 1]) * 1e0

    sampled_gt_CSI = gt_CSI.reshape(-1, 1).to("cuda")

    Ll1_CSI = sig2mse_complex(rendered_CSI, sampled_gt_CSI)

    tb_dict["l1"] = Ll1_CSI.item()

    rendered_abs = torch.sqrt(rendered_CSI.real ** 2 + rendered_CSI.imag ** 2 + 1e-12)

    gt_abs = torch.sqrt(sampled_gt_CSI.real ** 2 + sampled_gt_CSI.imag ** 2 + 1e-12)
    GT_abs_db = gt_abs
    L_abs = torch.abs(rendered_abs - gt_abs) * 1e0  # l2
    tb_dict["l_abs"] = L_abs.item()

    lambda_abs = 1
    lambda_csi = 50
    loss = L_abs * lambda_abs #+ Ll1_CSI * lambda_csi

    tb_dict["loss"] = loss.item()

    return loss, tb_dict


def sig2mse_complex(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    diff_real = torch.real(x) - torch.real(y)
    diff_imag = torch.imag(x) - torch.imag(y)
    return torch.mean(diff_real ** 2 + diff_imag ** 2)



def rendering_equation_RF_txGSrx(viewdirs, cov, incov, direct_signal, H_amp, H_phases, roughness, normals, reflection_coe, reflection_phase,
                                 visibility_precompute=None, incident_distances=None, incident_dirs=None):
    eps = 1e-20

    incident_dir_perGS = incident_dirs
    incident_areas = ellipsoid_projection_area(incident_dir_perGS, cov, incov)

    # compute spherical area denominator
    abs_frac = 4.0 * math.pi * (incident_distances.clamp(min=eps).pow_(2))
    incident_signals_pow = incident_areas.unsqueeze(-1) / abs_frac * visibility_precompute * direct_signal

    f_s, _ = Degli_Esposti_scattering(normals, viewdirs, incident_dir_perGS, roughness, BSDF_class='Degli-Esposti')
    pbr_a = f_s * torch.sqrt(incident_signals_pow + eps) * reflection_coe   # reflection_coe就定义为幅度
    pbr_p = reflection_phase.unsqueeze(-1) + H_phases  #torch.exp(1j * reflection_phase)
    dd = pbr_a.unsqueeze(-1).expand(-1, -1, H_phases.size(2)) * H_amp
    qq = pbr_p #.squeeze(-1)
    pbr = torch.cat((dd, qq), dim=1)

    extra_results = {
        "incident_dirs": incident_dir_perGS,
        "specular": pbr
    }

    return pbr, extra_results



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


def rendering_equation_RF(roughness, normals, viewdirs,  # 一
                       incidents, reflection_coe, reflection_phase, direct_light_env_light=None,
                       visibility_precompute=None, incident_dirs_precompute=None, incident_areas_precompute=None):

    incident_dirs, incident_areas = incident_dirs_precompute, incident_areas_precompute
    deg = int(np.sqrt(incidents.shape[1]) - 1)
    global_incident_lights = direct_light_env_light.direct_light(incident_dirs)
    local_incident_lights = eval_sh(deg, incidents.transpose(1, 2).view(-1, 1, 3, (deg + 1).pow_(2)),
                                    incident_dirs).clamp_min(0)  。

    incident_visibility = visibility_precompute
    global_incident_lights = global_incident_lights * incident_visibility
    incident_lights = local_incident_lights # + global_incident_lights

    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)

    f_s = Degli_Esposti_scattering(normals, viewdirs, incident_dirs, roughness, fresnel=0.04)

    transport = incident_lights * incident_areas * n_d_i
    CSI_s = ((f_s) * transport * reflection_coe * reflection_phase).mean(dim=-2)
    pbr = (f_s * transport).mean(dim=-2)  # (8)
    diffuse_light = transport.mean(dim=-2)

    extra_results = {
        "incident_dirs": incident_dirs,
        "incident_lights": incident_lights,
        "local_incident_lights": local_incident_lights,
        "global_incident_lights": global_incident_lights,
        "incident_visibility": incident_visibility,
        "diffuse_light": diffuse_light,
        "specular": CSI_s,  # 维度需要确认
    }

    return pbr, extra_results


def Degli_Esposti_scattering(normal, ws, wi, roughness, BSDF_class):

    if BSDF_class == "Degli-Esposti":
        WI = F.normalize(wi, dim=-1)
        N = F.normalize(normal, dim=-1)
        NoWI = torch.sum(N * WI, dim=-1, keepdim=True).clamp_(1e-6, 1)

        WS = F.normalize(ws, dim=-1)
        NoWS = torch.sum(WS * N, dim=-1, keepdim=True)
        N = N * NoWS.sign()
        NoWS = torch.sum(N * WS, dim=-1, keepdim=True).clamp_(1e-6, 1)

        R = WI - 2 * NoWI * N   #
        WSoR = torch.sum(WS * R, dim=-1, keepdim=True).clamp_(1e-6, 1)

        eps = 1e-6
        WSoR_safe = WSoR.clamp(-1.0 + eps, 1.0 - eps)
        base = (1.0 + WSoR_safe) * 0.5
        rough_safe = torch.clamp_min(roughness, 1e0)
        f_pow_out = torch.pow(base.clamp_min(eps), rough_safe)
        f_thetai_phii_thetas_phis_ = f_pow_out.clone()

        f_thetai_phii_thetas_phis_ *= NoWS.pow_(0.5)

        k_alphaR = 1 / (0.07937 * roughness + 0.1745)
        F_alphaR_wi = NoWI.pow_(0.5) * k_alphaR   # (16)

        beta = NoWI / F_alphaR_wi * f_thetai_phii_thetas_phis_
        alpha = np.pi
    else:
        beta = 1/np.pi
        alpha = np.pi

    return torch.sqrt(torch.clamp(beta, min=0.0) + 1e-20), alpha


def compute_channel_H(distances, freqs):

    distances = torch.as_tensor(distances, dtype=torch.float32)
    freqs = torch.as_tensor(freqs, dtype=torch.float32)

    # c = sc.c
    c = 299792458.0
    wavelength = c / freqs

    phase_shift_raw = 2 * torch.pi * distances[..., None] / wavelength[None, :]
    phase_shift = torch.fmod(phase_shift_raw, 2 * torch.pi)

    amp_decay = 1 / distances[..., None]

    return amp_decay, phase_shift


def render_neilf_RSSI_prediction(viewpoint_antenna: Antenna, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                 scaling_modifier=1.0, override_color=None, opt: OptimizationParams = False,
                 is_training=False, dict_params=None, **kwargs):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    results = render_view(viewpoint_antenna, pc, pipe, bg_color, is_training, dict_params,
                        use_trained_exp=False)

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_antenna, pc, results, opt, direct_light_env_light=dict_params['env_light'])
        results["tb_dict"] = tb_dict
        results["loss"] = loss

    return results






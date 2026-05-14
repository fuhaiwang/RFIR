import os
from typing import NamedTuple

import numpy as np
import torch.nn as nn
import torch
from utils.system_utils import Timing

try:
    from diff_gaussian_rasterization import _C
except Exception as e:
    from torch.utils.cpp_extension import load

    parent_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "diff_gaussian_rasterization")
    _C = load(
        name='diff_gaussian_rasterization',
        extra_cuda_cflags=["-I " + os.path.join(parent_dir, "third_party/glm/"), "-O3"],
        extra_cflags=["-O3"],
        sources=[
            os.path.join(parent_dir, "cuda_rasterizer/rasterizer_impl.cu"),
            os.path.join(parent_dir, "cuda_rasterizer/forward.cu"),
            os.path.join(parent_dir, "cuda_rasterizer/backward.cu"),
            os.path.join(parent_dir, "rasterize_points.cu"),
            os.path.join(parent_dir, "ext.cpp")],
        verbose=True)


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
        means3D,
        means2D,
        features,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        features,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            means3D,
            means2D,
            features,
            sh,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,  # 41
            means3D,
            features,  # my
            colors_precomp,  # rssi
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,  # 42
            cov3Ds_precomp,  # 8
            raster_settings.viewmatrix,  # 43
            raster_settings.projmatrix,  # 44
            raster_settings.tanfovx,  # 39
            raster_settings.tanfovy,  # 40
            raster_settings.image_height,  # 37
            raster_settings.image_width,  # 38
            sh,  # 15
            raster_settings.sh_degree,  # 45
            raster_settings.campos,  # 46
            raster_settings.prefiltered,  # 47
            raster_settings.antialiasing,  # 49
            raster_settings.debug  # 48
        )

        # Invoke C++/CUDA rasterizer  features
        num_rendered, color, feature, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)

        color1 = color.detach().cpu().numpy()

        feature1 = feature.detach().cpu().numpy()

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, features, scales, rotations, cov3Ds_precomp, radii, sh, opacities,
                              geomBuffer, binningBuffer, imgBuffer)  # 用于c
        return color, feature, radii, invdepths     # init中设计rendered_image, radii, depth_image = rasterizer()的

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_feature, _, grad_out_depth):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, features, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                features,
                radii,
                colors_precomp,
                opacities,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,
                grad_out_depth,
                grad_out_feature,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug)

        arg_names = [
            "bg", "means3D", "features", "radii", "colors_precomp", "opacities",
            "scales", "rotations", "scale_modifier", "cov3Ds_precomp",
            "viewmatrix", "projmatrix", "tanfovx", "tanfovy", "grad_out_color",
            "grad_out_depth", "grad_out_feature", "sh", "sh_degree", "campos",
            "geomBuffer", "num_rendered", "binningBuffer", "imgBuffer",
            "antialiasing", "debug"
        ]

        # Compute gradients for relevant tensors by invoking backward method
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_features, grad_cov3Ds_precomp, \
        grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(
            *args)

        for i, g in enumerate([grad_means3D, grad_means2D, grad_features, grad_sh,
                               grad_colors_precomp, grad_opacities, grad_scales,
                               grad_rotations, grad_cov3Ds_precomp]):
            if g is not None and (torch.isnan(g).any() or torch.isinf(g).any()):
                print(f"[NaN DETECTED in BACKWARD output {i}] shape={g.shape}")

        grads = (
            grad_means3D,
            grad_means2D,
            grad_features,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    antialiasing: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)

        return visible

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None, scales=None, rotations=None,
                cov3D_precomp=None, features=None):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
                (scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        if features is None:
            features = torch.empty_like(means3D[..., :0])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            features,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )





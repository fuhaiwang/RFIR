#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import torch
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_fn_dict
from torchvision.utils import save_image
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from scene.direct_light_map import DirectLightMap
import numpy as np
import matplotlib.pyplot as plt
import os


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, pbr_kwargs=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    if gaussians.use_pbr:
        roughness_path = os.path.join(model_path, name, "ours_{}".format(iteration), "roughness")
        reflection_coe_path = os.path.join(model_path, name, "ours_{}".format(iteration), "reflection_coe")
        reflection_phase_path = os.path.join(model_path, name, "ours_{}".format(iteration), "reflection_phase")

        makedirs(roughness_path, exist_ok=True)
        makedirs(reflection_coe_path, exist_ok=True)
        makedirs(reflection_phase_path, exist_ok=True)

    render_fn = render_fn_dict[args.type]
    gt_RFS = []
    Render_RFS = []
    indexs = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gaussians.update_visibility_RF_single(view)
        results = render_fn(view, gaussians, pipeline, background, dict_params=pbr_kwargs)
        N = view.original_image.size(0)
        center_idx = 1000

        gt_raw = view.original_image[center_idx, :].cuda()

        roughness_image = results["roughness"].squeeze(0).unsqueeze(0).repeat(3, 1, 1)
        reflection_coe = results["reflection_coe"].squeeze(0).unsqueeze(0).repeat(3, 1, 1)
        reflection_phase = results["reflection_phase"].squeeze(0).unsqueeze(0).repeat(3, 1, 1)
        aa = np.array(reflection_phase.cpu())
        bb = np.array(roughness_image.cpu())
        cc = np.array(reflection_coe.cpu())

        min_rp_val = reflection_phase.min()
        max_rp_val = reflection_phase.max()
        normalized_reflection_phase = (reflection_phase - min_rp_val) / (max_rp_val - min_rp_val)

        min_ro_val = reflection_phase.min()
        max_ro_val = reflection_phase.max()
        normalized_roughness_image = (reflection_phase - min_ro_val) / (max_ro_val - min_ro_val)

        min_rc_val = reflection_coe.min()
        max_rc_val = reflection_coe.max()
        normalized_reflection_coe = (reflection_coe - min_rc_val) / (max_rc_val - min_rc_val)

        if gaussians.use_pbr:
            save_image(normalized_roughness_image, os.path.join(roughness_path, '{0:05d}'.format(idx) + ".png"))
            save_image(normalized_reflection_coe, os.path.join(reflection_coe_path, '{0:05d}'.format(idx) + ".png"))
            save_image(normalized_reflection_phase, os.path.join(reflection_phase_path, '{0:05d}'.format(idx) + ".png"))

        gt_CSI = (gt_raw[0] + 1j * gt_raw[1])
        sampled_gt_CSI = gt_CSI.reshape(-1, 1).to("cuda")
        gt_abs = torch.sqrt(sampled_gt_CSI.real ** 2 + sampled_gt_CSI.imag ** 2 + 1e-12)
        gt_abs_db = 10 * torch.log10(gt_abs)
        gt_RFS.append(gt_abs_db)

        index = view.image_name
        Render_abs = torch.sqrt(results["render"].cpu()**2  + 1e-12 ).to("cuda") / args.power_scale
        Render_abs_db = 10 * torch.log10(Render_abs)
        Render_RFS.append(Render_abs_db)

        indexs.append(index)

    ids = np.array(indexs)
    save_RF(gt_RFS, os.path.join(gts_path, "GT_RCS_360.png"), ids, use_dB=True)
    save_RF_polar(gt_RFS, os.path.join(gts_path, "GT_RCS_360_polor.png"), ids, use_dB=True)

    save_RF(Render_RFS, os.path.join(render_path, "Render_RCS_360.png"), ids, use_dB=True)
    save_RF_polar(Render_RFS, os.path.join(render_path, "Render_RCS_360_polor.png"), ids, use_dB=True)



def save_RF(RCS_list, save_path, indexs, use_dB=False):
    if isinstance(RCS_list[0], torch.Tensor):  # list of tensors
        RCS_list = torch.stack(RCS_list).detach().cpu().numpy()
    else:
        RCS_list = np.array(RCS_list)


    RCS_list = np.squeeze(np.squeeze(RCS_list))

    sorted_idx = np.argsort(indexs)
    indexs_sorted = indexs[sorted_idx]
    RCS_sorted = RCS_list[sorted_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(indexs_sorted, RCS_sorted, marker='o', linestyle='-')
    plt.xlabel("Angle (degree)")
    plt.ylabel("RCS (dBsm)")
    plt.title("RCS vs Angle")
    plt.grid(True)
    plt.show()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_RF_polar(RCS_list, save_path, indexs, use_dB=False):
    if isinstance(RCS_list[0], torch.Tensor):  # list of tensors
        RCS_list = torch.stack(RCS_list).detach().cpu().numpy()
    else:
        RCS_list = np.array(RCS_list)


    RCS_list = np.squeeze(np.squeeze(RCS_list))

    sorted_idx = np.argsort(indexs)
    indexs_sorted = indexs[sorted_idx]
    RCS_sorted = RCS_list[sorted_idx]

    theta = np.deg2rad(indexs_sorted)

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(theta, RCS_sorted, marker='o', linestyle='-')

    ax.set_title("RCS Polar Plot", va='bottom')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def render_sets(dataset: ModelParams, pipeline: PipelineParams, skip_train: bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
        scene = Scene(dataset, gaussians, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if args.checkpoint:
            print("Create Gaussians from checkpoint {}".format(args.checkpoint))
            iteration = gaussians.create_from_RF_ckpt(args.checkpoint, restore_optimizer=True)
        elif scene.loaded_iter:
            gaussians.load_ply(os.path.join(dataset.model_path,
                                            "point_cloud",
                                            "iteration_" + str(scene.loaded_iter),
                                            "point_cloud.ply"))
            iteration = scene.loaded_iter
        else:
            gaussians.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)
            iteration = scene.loaded_iter

        pbr_kwargs = dict()
        if iteration is not None and gaussians.use_pbr:
            pbr_kwargs['sample_num'] = args.sample_num
            print("Using global incident light for regularization.")
            direct_env_light = DirectLightMap(args.env_resolution)
            
            if args.checkpoint:
                env_checkpoint = os.path.dirname(args.checkpoint) + "/env_light_" + os.path.basename(args.checkpoint)
                print("Trying to load global incident light from ", env_checkpoint)
                if os.path.exists(env_checkpoint):
                    direct_env_light.create_from_ckpt(env_checkpoint, restore_optimizer=True)
                    print("Successfully loaded!")
                else:
                    print("Failed to load!")
                pbr_kwargs["env_light"] = direct_env_light

        if not skip_train:
             render_set(dataset.model_path, "train", iteration, scene.getTrainCameras(), gaussians, pipeline, background, pbr_kwargs)

        if not skip_test:
             render_set(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians, pipeline, background, pbr_kwargs)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('-t', '--type', choices=['render', 'normal', 'render_RF'], default='render_RF')
    parser.add_argument("--power_scale", type=float, default=None)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    torch.autograd.set_detect_anomaly(True)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    safe_state(args.quiet)

    render_sets(model.extract(args), pipeline.extract(args), args.skip_train, args.skip_test)








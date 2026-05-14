import torch
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_fn_dict
from torchvision.utils import save_image
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationDeformParams
from scene import Scene, GaussianModel
from scene.direct_light_map import DirectLightMap
import numpy as np
import matplotlib.pyplot as plt
import os
from scene.frequency_expansion_model import FrequencyExpandModel
from random import randint
import time
import random

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
    print(f" Photo: {save_path}")


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
    print(f" Polar saving: {save_path}")


def render_set(model_path, name, views, gaussians, opt, pipeline, background, pbr_kwargs=None):
    iteration = args.RF_iterations

    for idx, view in enumerate(tqdm(views, desc="update visibility progress for RF single")):
        gaussians.update_visibility_RF_single(view)

    N = gaussians.get_xyz.shape[0]
    center_frequency = 5.8e9
    bw = 160e6
    frequency = torch.linspace(
        center_frequency - bw / 2,
        center_frequency + bw / 2,
        steps=8192,
        device=gaussians.get_xyz.device
    ).unsqueeze(0)

    frequency_idx_ = frequency[:, 192:8192:4]
    render_fn = render_fn_dict[args.type2]

    freq_deform = FrequencyExpandModel(freq_dim=1)
    freq_deform.train_setting(opt)    # add 

    resume_iter = 0
    resume_ckpt_path = os.path.join(model_path, "wideband_deform", args.freq_checkpoint)
    if os.path.exists(resume_ckpt_path):
        print(f"[INFO] Resuming training from {resume_ckpt_path}")
        resume_iter = freq_deform.load_weights(resume_ckpt_path)

    first_iter = resume_iter
    progress_bar = tqdm(
        range(first_iter + 1, iteration + 1),
        desc="Training progress",
        initial=first_iter,
        total=iteration,
        position=0
    )

    viewpoint_stack = None
    sum_loss = 0
    all_loss = []
    view_counter= 0
    freq_counter = 0
    freq_idx = 0

    for iteration in progress_bar:
        if not viewpoint_stack:
            viewpoint_stack = views.copy()
        view_counter += 1
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        batch_size = 20
        for freq_idx in range(0, 2000//batch_size):  

            freq_counter += 1
            if freq_idx  % 10 == 0:
                print(freq_idx)
            frequency_idx = frequency_idx_[:,freq_idx*batch_size: (freq_idx+1)*batch_size]

            pbr_kwargs["freq_idx"] = freq_idx 
            frequency_input = frequency_idx.expand(N, -1)
            d_reflection_coe, d_reflection_phase, d_roughness = freq_deform.step(gaussians.get_xyz.detach(), frequency_input)

            render_pkg = render_fn(viewpoint_cam, gaussians, pipeline, background, d_reflection_coe, d_reflection_phase,
                                   d_roughness, batch_size, frequency_idx, opt, is_training=True, dict_params=pbr_kwargs, iteration=iteration)
            loss = render_pkg["loss"]
            loss.backward()
   
        with torch.no_grad():
            freq_deform.optimizer.step()  # add
            freq_deform.optimizer.zero_grad()  # add
            freq_deform.update_learning_rate(iteration)
            sum_loss += loss.item()
            all_loss.append(loss.item())


        if iteration % args.checkpoint_interval == 0 or iteration == args.RF_iterations:
            tqdm.write("\n[ITER {}] Saving Checkpoint".format(iteration))
            freq_deform.save_weights(model_path, iteration, freq_idx)
            mean_loss = sum_loss/args.checkpoint_interval
            log_path = os.path.join(model_path, "wideband_deform/iteration_{}/loss_log.txt".format(iteration))
            with open(log_path, "w") as f:
                f.write("Loss record\n")

            with open(log_path, "a") as f:
                f.write(f" Iter {iteration}, All loss {all_loss} Mean loss {mean_loss:.6f} \n")

            print(mean_loss) 
            sum_loss = 0   
            all_loss = []


def wideband_render_sets(dataset: ModelParams, opt: OptimizationDeformParams, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    
    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type1)

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
    pbr_kwargs['power_scale'] = args.power_scale
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
        render_set(dataset.model_path, "train_wideband", scene.getTrainCameras(), gaussians, opt, pipeline, background, pbr_kwargs)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationDeformParams(parser)  
    parser.add_argument("--RF_iterations", default=500, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=200)
    parser.add_argument('-t1', '--type1', choices=['render', 'normal', 'render_RF'], default='render_RF')
    parser.add_argument('-t2', '--type2', choices=["render_RF_bb"], default='render_RF_bb')
    parser.add_argument("-c", "--checkpoint", type=str, default='output/real_image/H_v2/render_RF_v7/chkpnt32000.pth')
    parser.add_argument("--power_scale", type=float, default=1)
    parser.add_argument("-fc", "--freq_checkpoint", type=str, default='iteration_1000/wideband_deform.pth')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    wideband_render_sets(model.extract(args), op.extract(args), pipeline.extract(args), args.skip_train, args.skip_test)




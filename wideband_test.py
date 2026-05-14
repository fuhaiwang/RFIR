import torch
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_fn_dict
from torchvision.utils import save_image
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from scene import Scene, GaussianModel
from scene.direct_light_map import DirectLightMap
import numpy as np
import matplotlib.pyplot as plt
import os
from scene import FrequencyExpandModel
# from arguments import ModelParams, PipelineParams, OptimizationParams
from random import randint
import matplotlib.pyplot as plt
import json


def render_set(model_path, name, views, gaussians, opt, pipeline, background, frequency_idx_, pbr_kwargs=None):
    render_fn = render_fn_dict[args.type2]


    indexs = []

    with torch.no_grad():

        freq_deform = FrequencyExpandModel(freq_dim=1)
        resume_ckpt_path = os.path.join(model_path, "wideband_deform", args.freq_checkpoint)

        iters = freq_deform.load_weights(resume_ckpt_path)
        Render_RFS_all_views = []

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            
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

            Render_RFS = []

            batch_size = 20
            for freq_idx in range(0, 2000//batch_size):
                pbr_kwargs["freq_idx"] = freq_idx
                frequency_idx = frequency_idx_[:, freq_idx*batch_size: (freq_idx+1)*batch_size]

                frequency_input = frequency_idx.expand(N, -1)
                d_reflection_coe, d_reflection_phase, d_roughness = freq_deform.step(gaussians.get_xyz.detach(), frequency_input)

                render_pkg = render_fn(view, gaussians, pipeline, background, d_reflection_coe, d_reflection_phase,
                                    d_roughness, batch_size, frequency_idx, opt, is_training=False, dict_params=pbr_kwargs,
                                    iteration=idx)

                rendered_CSI_freq = torch.abs(render_pkg["render"].cpu()).to("cuda") / args.power_scale 

                Render_RFS.append(rendered_CSI_freq) # 纯信号的abs，没有取dB

            Render_RFS = torch.cat(Render_RFS, dim=0) 
            Render_RFS_all_views.append(Render_RFS)

            index = view.image_name
            indexs.append(index)
        ids = np.array(indexs)

        Render_RFS_all_views = torch.stack(Render_RFS_all_views, dim=0)

    return Render_RFS_all_views, ids


def plot_polar_distribution(data, cmap="viridis", title="RCS", save_path="polar_rcs.png"):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    vmin, vmax = np.percentile(data, [1, 99.99])
    data_clip = np.clip(data, vmin, vmax)
    data_norm = (data_clip - vmin) / (vmax - vmin)

    N_theta, N_r = data_norm.shape

    theta = np.linspace(0, 2 * np.pi, N_theta)
    r = np.linspace(0, 1, N_r)

    Theta, R = np.meshgrid(theta, r, indexing='ij')

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.pcolormesh(X, Y, data_norm, shading='auto', cmap=cmap)
    plt.axis("equal")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    r_text = 1.02
    for deg in range(0, 360, 45):
        rad = np.deg2rad(deg)
        x = r_text * np.cos(rad)
        y = r_text * np.sin(rad)
        ax.text(1.05*x, 1.05*y, f"{deg}°", ha='center', va='center', fontsize=10)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  



def save_RF_polar(RCS_list, path, file_name, indexs, use_dB=False):
    save_path = os.path.join(path, file_name)

    if isinstance(RCS_list[0], torch.Tensor):
        RCS_list = torch.stack(RCS_list).detach().cpu().numpy()
    else:
        RCS_list = np.array(RCS_list)

    RCS_list = np.squeeze(np.squeeze(RCS_list))

    sorted_idx = np.argsort(indexs)
    indexs_sorted = indexs[sorted_idx]
    Render_abs_db = 10 * np.log10(RCS_list)
    RCS_sorted = Render_abs_db

    data_to_save = {
        "angles": indexs_sorted.tolist(),
        "rcs_values": RCS_sorted.tolist()
    }
    with open(os.path.join(path,'RCS_data.json'), 'w') as f:
        json.dump(data_to_save, f, indent=4)

    theta = np.deg2rad(indexs_sorted)

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(theta, RCS_sorted, marker='o', linestyle='-')
    ax.set_rlim(-20, -5)

    ax.set_title("RCS Polar Plot", va='bottom')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"save path: {save_path}")

def save_data_to_json(data, json_path="rcs_data.json"):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    data_list = data.tolist()

    with open(json_path, "w") as f:
        json.dump(data_list, f, indent=4)


def wideband_render_sets(dataset: ModelParams, opt: OptimizationParams, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    
    save_dir = os.path.join(dataset.model_path, "predicted_RSSI")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"creating dir: {save_dir}")

    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type1)
    
    scene = Scene(dataset, gaussians, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        iteration = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)
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
        gaussians.update_visibility(args.sample_num)

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


    center_frequency = 5.8e9
    bw = 160e6
    frequency = torch.linspace(
        center_frequency - bw / 2,
        center_frequency + bw / 2,
        steps=8192,
        device=gaussians.get_xyz.device
    ).unsqueeze(0)
    frequency_idx_ = frequency[:, 192:8192:4]

    result_CSI, ids = render_set(dataset.model_path, "test_wideband", scene.getTestCameras(), gaussians, opt, pipeline, background, frequency_idx_, pbr_kwargs)
    Rendered_RCS = result_CSI.cpu().numpy()
    plot_polar_distribution(Rendered_RCS, cmap="plasma", title="Rendered_RCS", save_path=os.path.join(dataset.model_path, "predicted_RSSI/Rendered_RCS.png"))
    save_data_to_json(Rendered_RCS, os.path.join(dataset.model_path,"predicted_RSSI/Rendered_rcs_data.json"))

    RCS_all_ = []
    viewpoint_antenna = scene.getTestCameras()
    for idx, view in enumerate(tqdm(viewpoint_antenna, desc="Rendering progress")):
        gt_raw_all = view.original_image.cuda()
        RSSI = torch.norm(gt_raw_all, dim=1)
        RCS_all_.append(RSSI)

    RCS_all = torch.stack(RCS_all_, dim=0)
    GT_RCS = RCS_all.cpu().numpy()

    plot_polar_distribution(GT_RCS, cmap="plasma", title="GT_RCS", save_path=os.path.join(dataset.model_path, "predicted_RSSI/GT_RCS.png"))  #[:, 0:1000]

    save_data_to_json(GT_RCS, os.path.join(dataset.model_path,"predicted_RSSI/GT_rcs_data.json"))



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)  # 30000
    parser.add_argument("--RF_iterations", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument('-t1', '--type1', choices=['render', 'normal', 'render_RF'], default='render_RF')
    parser.add_argument('-t2', '--type2', choices=["render_RF_bb"], default='render_RF_bb')
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--power_scale", type=float, default=1)
    parser.add_argument("-fc", "--freq_checkpoint", type=str, default='iteration_1400/wideband_deform.pth')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    wideband_render_sets(model.extract(args), op.extract(args), pipeline.extract(args), args.skip_train, args.skip_test)






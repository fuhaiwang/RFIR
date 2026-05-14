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
import json
from scene import Renderer_LoS_NLoS


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, renderer_los_nlos, pbr_kwargs=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
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
    Rx_pos_all = []
    Tx_pos_all = []
    indexs = []

    flag_gt = False

    gt_RFS = []
    Render_RFS = []
    gt_RFS_dB = []
    Render_RFS_dB = []
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        Rx_pos = view.Rx_pos.detach().cpu().numpy()
        Tx_pos = view.Tx_pos.detach().cpu().numpy()

        if not flag_gt:
            gaussians.update_visibility_RF_single(view)
            results = render_fn(view, gaussians, pipeline, background, renderer_los_nlos, dict_params=pbr_kwargs)
            FREQ_MAPPING = {
                '2.4G': 0,
            }
            gt_raw = view.original_image[FREQ_MAPPING[args.freq], :].cuda()

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

            gt_CSI = (gt_raw[0] + 1j * gt_raw[1])
            sampled_gt_CSI = gt_CSI.reshape(-1, 1).to("cuda")
            gt_abs = torch.sqrt(sampled_gt_CSI.real ** 2 + sampled_gt_CSI.imag ** 2 + 1e-12)
            gt_RFS.append(gt_abs)

            gt_abs_db = 10 * torch.log10(gt_abs) 
            gt_RFS_dB.append(gt_abs_db)

            index = view.image_name
            Render_abs = torch.abs(results["render"].cpu() * pbr_kwargs["power_scalar"]).to("cuda")
            Render_RFS.append(Render_abs)

            Render_abs_db = 10 * torch.log10(Render_abs)

            Render_RFS_dB.append(Render_abs_db)
            indexs.append(index)

        Rx_pos_all.append(Rx_pos)
        Tx_pos_all.append(Tx_pos)

    save_RF_dB(torch.tensor(gt_RFS_dB).cpu().detach(), gts_path, "GT_RCS_360_dB.png", use_dB=True)
    save_RF(torch.tensor(gt_RFS).cpu().detach(), gts_path, "GT_RCS_360.png", use_dB=True)
    
    plot_topdown_heatmap2(Tx_pos, np.array(Rx_pos_all), torch.tensor(gt_RFS_dB).cpu().detach(), grid_size=300, save_path=os.path.join(model_path, name, "ours_{}".format(iteration)),
    name="rssi_topdown_heatmap_gt.png")

    save_RF_dB(torch.tensor(Render_RFS_dB).cpu().detach(), render_path, "Render_RCS_360_dB.png", use_dB=True)
    plot_topdown_heatmap2(Tx_pos, np.array(Rx_pos_all), torch.tensor(Render_RFS_dB).cpu().detach(), grid_size=300, save_path=os.path.join(model_path, name, "ours_{}".format(iteration)),name="rssi_topdown_heatmap_render.png")



def save_RF_dB(RCS_list, path, file_name, use_dB=False):

    save_path = os.path.join(path, file_name)
    RCS_list = np.array(RCS_list)

    RCS_list = np.squeeze(np.squeeze(RCS_list))
    RCS_sorted = RCS_list

    data_to_save = {
        "rcs_values": RCS_sorted.tolist()
    }
    with open(os.path.join(path, 'RCS_data_dB.json'), 'w') as f:
        json.dump(data_to_save, f, indent=4)

    plt.figure(figsize=(8, 5))
    plt.plot(RCS_sorted, marker='o', linestyle='-')
    plt.xlabel("Angle (degree)")
    plt.ylabel("RCS (dBsm)")
    plt.title("RCS vs Angle")
    plt.grid(True)
    plt.show()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved: {save_path}")


def save_RF(RCS_list, path, file_name, use_dB=False):

    save_path = os.path.join(path, file_name)
    RCS_list = np.array(RCS_list)

    RCS_list = np.squeeze(np.squeeze(RCS_list))
    RCS_sorted = RCS_list

    data_to_save = {
        "rcs_values": RCS_sorted.tolist()
    }
    with open(os.path.join(path, 'RCS_data.json'), 'w') as f:
        json.dump(data_to_save, f, indent=4)

    plt.figure(figsize=(8, 5))
    plt.plot(RCS_sorted, marker='o', linestyle='-')
    plt.xlabel("Angle (degree)")
    plt.ylabel("RCS")
    plt.title("RCS vs Angle")
    plt.grid(True)
    plt.show()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" saved: {save_path}")


def plot_topdown_heatmap2(Tx_pos, Rx_pos_all, RSSI, grid_size=200, method='linear', save_path=None, name=None):
    from scipy.interpolate import griddata
    x = Rx_pos_all[:, 0]
    z = Rx_pos_all[:, 1]
    y = Rx_pos_all[:, 2]

    mask = np.isclose(z, 2.0, atol=1e-6)
    indices = np.where(mask)[0]

    Rx_pos_y1 = Rx_pos_all[indices]
    RSSI = np.array(RSSI, dtype=float).flatten()
    RSSI_y1 = RSSI[indices]

    x_ = Rx_pos_y1[:, 0]
    y_ = Rx_pos_y1[:, 2]
    rssi_dB = RSSI_y1

    x_i = np.linspace(x_.min(), x_.max(), grid_size)
    y_i = np.linspace(y_.min(), y_.max(), grid_size)
    x_i, y_i = np.meshgrid(x_i, y_i)

    zi = griddata((x_, y_), rssi_dB, (x_i, y_i), method=method, fill_value=np.nan)
    mask_a = (x_i >= -3.5) & (x_i <= 4.5) & (y_i >= -5) & (y_i <= 5)
    mask_b = (x_i >= 4.5) & (x_i <= 6.5) & (y_i >= -23) & (y_i <= 20)
    combined_mask = mask_a | mask_b

    zi[~combined_mask] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6))
    z_min = np.nanmin(-40)
    z_max = np.nanmax(-12)

    im = ax.pcolormesh(x_i, y_i, zi, cmap='viridis', shading='auto',
                       vmin=z_min, vmax=z_max, alpha=0.9)

    ax.scatter(Tx_pos[0], Tx_pos[2], c='red', marker='*', s=50,
               edgecolors='black', linewidths=0.2, label='Tx', zorder=5)

    ax.set_aspect('equal', adjustable='box')

    ax.set_title("RSSI Heatmap at y = 2 m", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("X position (m)", fontsize=12)
    ax.set_ylabel("Z position (m)", fontsize=12)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("RSSI (dB)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.set_facecolor('#f0f0f0')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    if save_path:
        if save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = save_path
            dir_name = os.path.dirname(full_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
        else:
            os.makedirs(save_path, exist_ok=True)
            file_name = name if name is not None else "heatmap.png"
            full_path = os.path.join(save_path, file_name)

        plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"saved: {full_path}")

    return fig, ax


def testing(dataset: ModelParams, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
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

        pbr_kwargs["freq"] = args.freq 
        pbr_kwargs["power_scalar"] = args.power_scalar 

        renderer_los_nlos = Renderer_LoS_NLoS()
        checkpoint_name = f"los_nlos_weights_{iteration}.pth"
        full_checkpoint_path = os.path.join(dataset.model_path, checkpoint_name)
        renderer_los_nlos.load_weights(full_checkpoint_path)

        # if not skip_train:
        #     render_set(dataset.model_path, "train", iteration, scene.getTrainCameras(), gaussians, pipeline, background, renderer_los_nlos, pbr_kwargs)

        if not skip_train:
            render_set(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians, pipeline, background, renderer_los_nlos, pbr_kwargs)





if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('-t', '--type', choices=['render', 'normal', 'render_RF', 'render_RF_RSSI_prediction'], default='render_RF_RSSI_prediction')
    parser.add_argument("-freq", "--freq", choices=['2.4G', '5.8G'], default=None)
    parser.add_argument("-power_scalar", "--power_scalar", default=1.0, type=float)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    torch.autograd.set_detect_anomaly(True)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    safe_state(args.quiet)

    testing(model.extract(args), pipeline.extract(args), args.skip_train, args.skip_test)














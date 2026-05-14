import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from collections import defaultdict
from random import randint
from utils.loss_utils import ssim
from gaussian_renderer import render_fn_dict
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr, visualize_depth
from utils.system_utils import prepare_output_and_logger
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
# from gui import GUI
from scene.direct_light_map import DirectLightMap
from utils.graphics_utils import rgb_to_srgb
from torchvision.utils import save_image, make_grid
from lpipsPyTorch import lpips
import matplotlib.pyplot as plt
from scene import FrequencyExpandModel


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, is_pbr=False):
    torch.autograd.set_detect_anomaly(True)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    """  
    Setup Gaussians 
    """
    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
    scene = Scene(dataset, gaussians, shuffle=True)
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)
    elif scene.loaded_iter:
        gaussians.load_ply(os.path.join(dataset.model_path,
                                        "point_cloud",
                                        "iteration_" + str(scene.loaded_iter),
                                        "point_cloud.ply"))
    else:
        gaussians.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)
    gaussians.training_setup(opt)

    """
    Setup PBR components
    """
    pbr_kwargs = dict()
    if is_pbr:
        pbr_kwargs['sample_num'] = pipe.sample_num
        print("Using global incident light for regularization.")
        direct_env_light = DirectLightMap(dataset.env_resolution, opt.light_init)
        
        if args.checkpoint:
            env_checkpoint = os.path.dirname(args.checkpoint) + "/env_light_" + os.path.basename(args.checkpoint)
            print("Trying to load global incident light from ", env_checkpoint)
            if os.path.exists(env_checkpoint):
                direct_env_light.create_from_ckpt(env_checkpoint, restore_optimizer=True)
                print("Successfully loaded!")
            else:
                print("Failed to load!")

            direct_env_light.training_setup(opt)
            pbr_kwargs["env_light"] = direct_env_light

    """ Prepare render function and bg"""
    render_fn = render_fn_dict[args.type]  # 渲染方法
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    """ GUI """
    windows = None
    """""""""""""""" Training!!!!!!!!!!!!!!!!!!! """""""""""""""
    viewpoint_stack = None
    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(range(first_iter + 1, opt.iterations + 1), desc="Training progress",
                        initial=first_iter, total=opt.iterations)
    total_loss = 0.0
    loss_count = 0

    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)
        # print(iteration)
        if windows is not None:
            windows.render()

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if is_pbr:
            gaussians.update_visibility_RF_single(viewpoint_cam)

        # Render
        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        pbr_kwargs["iteration"] = iteration - first_iter
        pbr_kwargs["freq"] = args.freq
        pbr_kwargs["power_scale"] = args.power_scale
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                                opt=opt, is_training=True, dict_params=pbr_kwargs, iteration=iteration)

        viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        tb_dict = render_pkg["tb_dict"]
        current_loss = render_pkg["loss"]
        total_loss += current_loss.item()
        loss_count += 1
        
        loss = current_loss
        loss.backward() 

        if iteration % 20 == 0:
            avg_loss = total_loss / loss_count
            print(f"\n[Iteration {iteration}] Average Loss (last 20 iters): {avg_loss:.6f}")

            total_loss = 0.0
            loss_count = 0

        del loss
        torch.cuda.empty_cache()

        with torch.no_grad():
            if pipe.save_training_vis:
                if not is_pbr:
                    save_training_vis(viewpoint_cam, gaussians, background, render_fn,
                                    pipe, opt, first_iter, iteration, pbr_kwargs)
            # Progress bar
            pbar_dict = {"num": gaussians.get_xyz.shape[0]}
            if is_pbr:
                pbar_dict["light_mean"] = direct_env_light.get_env.mean().item()
                pbar_dict["env"] = direct_env_light.H


            for k in tb_dict:
                if k in ["psnr", "psnr_pbr"]:
                    ema_dict_for_log[k] = 0.4 * tb_dict[k] + 0.6 * ema_dict_for_log[k]
                    pbar_dict[k] = f"{ema_dict_for_log[k]:.{7}f}"
            progress_bar.set_postfix(pbar_dict)

            # Log and save
            if is_pbr:
                training_report_RF(tb_writer, iteration, tb_dict,
                                scene, render_fn, pipe=pipe,
                                bg_color=background, dict_params=pbr_kwargs)
            else:
                training_report(tb_writer, iteration, tb_dict,
                                scene, render_fn, pipe=pipe,
                                bg_color=background, dict_params=pbr_kwargs)

            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, render_pkg['weights'])
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                        radii[visibility_filter])

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    densify_grad_normal_threshold = opt.densify_grad_normal_threshold if iteration > opt.normal_densify_from_iter else 99999
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold,
                                                densify_grad_normal_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                
                if iteration % 50 == 0:
                    torch.cuda.empty_cache()

            # Optimizer step
            gaussians.step() 
            for component in pbr_kwargs.values():
                try:
                    component.step()
                except:
                    pass

            # save checkpoints
            if iteration % args.save_interval == 0 or iteration == args.iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            if iteration % args.checkpoint_interval == 0 or iteration == args.iterations:
                torch.save((gaussians.capture(), iteration),
                           os.path.join(scene.model_path, "chkpnt" + str(iteration) + ".pth"))
                torch.cuda.empty_cache()

                for com_name, component in pbr_kwargs.items():
                    try:
                        torch.save((component.capture(), iteration),
                                   os.path.join(scene.model_path, f"{com_name}_chkpnt" + str(iteration) + ".pth"))
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    except:
                        pass

                    print("[ITER {}] Saving {} Checkpoint".format(iteration, com_name))

    if dataset.eval:
        if is_pbr:
            eval_RF_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs)# add
        else:
            eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs)


def training_report(tb_writer, iteration, tb_dict, scene: Scene, renderFunc, pipe,
                    bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None,
                    opt: OptimizationParams = None, is_training=False, **kwargs):
    if tb_writer:
        for key in tb_dict:
            tb_writer.add_scalar(f'train_loss_patches/{key}', tb_dict[key], iteration)

    # Report test and samples of training set
    if iteration % args.test_interval == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                psnr_pbr_test = 0.0
                for idx, viewpoint in enumerate(
                        tqdm(config['cameras'], desc="Evaluating " + config['name'], leave=False)):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, pipe, bg_color,
                                            scaling_modifier, override_color, opt, is_training,
                                            **kwargs)

                    image = render_pkg["render"]
                    gt_image = viewpoint.original_image.cuda()

                    opacity = torch.clamp(render_pkg["opacity"], 0.0, 1.0)
                    depth = render_pkg["depth"]
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    normal = torch.clamp(
                        render_pkg.get("normal", torch.zeros_like(image)) / 2 + 0.5 * opacity, 0.0, 1.0)

                    # BRDF
                    base_color = torch.clamp(render_pkg.get("base_color", torch.zeros_like(image)), 0.0, 1.0)
                    roughness = torch.clamp(render_pkg.get("roughness", torch.zeros_like(depth)), 0.0, 1.0)
                    image_pbr = render_pkg.get("pbr", torch.zeros_like(image))

                    grid = torchvision.utils.make_grid(
                        torch.stack([image, image_pbr, gt_image,
                                     opacity.repeat(3, 1, 1), depth.repeat(3, 1, 1), normal,
                                     base_color, roughness.repeat(3, 1, 1)], dim=0), nrow=3)

                    if tb_writer and (idx < 2):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             grid[None], global_step=iteration)

                    l1_test += F.l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    psnr_pbr_test += psnr(image_pbr, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                psnr_pbr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} PSNR_PBR {}".format(iteration, config['name'], l1_test,
                                                                                    psnr_test, psnr_pbr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_pbr', psnr_pbr_test, iteration)
                if iteration == args.iterations:
                    with open(os.path.join(args.model_path, config['name'] + "_loss.txt"), 'w') as f:
                        f.write("L1 {} PSNR {} PSNR_PBR {}".format(l1_test, psnr_test, psnr_pbr_test))

        torch.cuda.empty_cache()


def training_report_RF(tb_writer, iteration, tb_dict, scene: Scene, renderFunc, pipe,
                    bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None,
                    opt: OptimizationParams = None, is_training=False, **kwargs):
    if tb_writer:
        for key in tb_dict:
            tb_writer.add_scalar(f'train_loss_patches/{key}', tb_dict[key], iteration)

    # Report test and samples of training set
    if iteration % args.test_interval == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l2_test = 0.0
                for idx, viewpoint in enumerate(
                        tqdm(config['cameras'], desc="Evaluating " + config['name'], leave=False)):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, pipe, bg_color,
                                            scaling_modifier, override_color, opt, is_training,
                                            **kwargs)

                    image = render_pkg["render"].unsqueeze(-1).to("cuda")
                    gt_raw = viewpoint.original_image.cuda()
                    gt_CSI = (gt_raw[:, 0] + 1j * gt_raw[:, 1])   # 数值都扩大一些 一
                    indices = torch.linspace(0, gt_CSI.size(0) - 1, steps=1, dtype=torch.long)
                    sampled_gt_CSI = gt_CSI[indices].reshape(-1, 1).to("cuda")
                    l2_test += sig2mse_complex(image, sampled_gt_CSI)

                l2_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {}".format(iteration, config['name'], l2_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l2_test, iteration)
                if iteration == args.iterations:
                    with open(os.path.join(args.model_path, config['name'] + "_loss.txt"), 'w') as f:
                        f.write("L1 {} ".format(l2_test))

        torch.cuda.empty_cache()


def sig2mse_complex(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    diff_real = torch.real(x) - torch.real(y)
    diff_imag = torch.imag(x) - torch.imag(y)
    return torch.mean(diff_real ** 2 + diff_imag ** 2)


def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, first_iter, iteration, pbr_kwargs):
    os.makedirs(os.path.join(args.model_path, "visualize"), exist_ok=True)
    with torch.no_grad():
        if iteration % pipe.save_training_vis_iteration == 0 or iteration == first_iter + 1:
            render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                                   opt=opt, is_training=False, dict_params=pbr_kwargs)

            visualization_list = []
            RCS_list = []
            if is_pbr:
                visualization_list.extend([
                    render_pkg["roughness"].repeat(3, 1, 1),
                    render_pkg["reflection_coe"].repeat(3, 1, 1),
                    render_pkg["reflection_phase"].repeat(3, 1, 1),
                ])
                grid = torch.stack(visualization_list, dim=0)
                grid = make_grid(grid, nrow=4)
                scale = grid.shape[-2] / 800
                grid = F.interpolate(grid[None], (int(grid.shape[-2] / scale), int(grid.shape[-1] / scale)))[0]
                x = render_pkg["render"]
                RCS_list.extend([
                    x.real ** 2 + x.imag ** 2
                ])
            else:
                visualization_list.extend([
                    render_pkg["render"],
                    viewpoint_cam.original_image.cuda(),
                    visualize_depth(render_pkg["depth"]),
                    (render_pkg["depth_var"] / 0.001).clamp_max(1).repeat(3, 1, 1),
                    render_pkg["opacity"].repeat(3, 1, 1),
                    render_pkg["normal"] * 0.5 + 0.5,
                    render_pkg["pseudo_normal"] * 0.5 + 0.5,
                ])
                grid = torch.stack(visualization_list, dim=0)
                grid = make_grid(grid, nrow=4)
                scale = grid.shape[-2] / 800
                grid = F.interpolate(grid[None], (int(grid.shape[-2]/scale), int(grid.shape[-1]/scale)))[0]
                save_image(grid, os.path.join(args.model_path, "visualize", f"{iteration:06d}.png"))
                depth_rgb = visualize_depth(render_pkg["depth"])        # [3,H,W]
                alpha = render_pkg["opacity"].clamp(0, 1)               # [1,H,W]
                depth_rgba = torch.cat([depth_rgb, alpha], dim=0)       # [4,H,W]
                save_image(
                    depth_rgba,
                    os.path.join(args.model_path, "visualize", f"{iteration:06d}_depth_rgba.png")
                )

                pseudo_rgb = render_pkg["pseudo_normal"] * 0.5 + 0.5
                alpha = render_pkg["opacity"].clamp(0, 1)
                pseudo_rgba = torch.cat([pseudo_rgb, alpha], dim=0)
                save_image(
                    pseudo_rgba,
                    os.path.join(args.model_path, "visualize", f"{iteration:06d}_pseudo_normal_rgba.png")
                )

                normal_rgb = render_pkg["normal"] * 0.5 + 0.5
                alpha = render_pkg["opacity"].clamp(0, 1)
                normal_rgba = torch.cat([normal_rgb, alpha], dim=0)
                save_image(
                    normal_rgba,
                    os.path.join(args.model_path, "visualize", f"{iteration:06d}_normal_rgba.png")
                )

                orig_img = viewpoint_cam.original_image.cuda()
                save_image(
                    orig_img,
                    os.path.join(args.model_path, "visualize", f"{iteration:06d}_input.png")
                )


def save_RF(RCS_list, save_path, use_dB=False):
    if isinstance(RCS_list[0], torch.Tensor):  # list of tensors
        RCS_list = torch.stack(RCS_list).detach().cpu().numpy()
    else:
        RCS_list = np.array(RCS_list)

    if use_dB:
        RCS_list = 10 * np.log10(np.maximum(RCS_list, 1e-12))

    theta = np.linspace(0, 2*np.pi, len(RCS_list), endpoint=False)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(theta, RCS_list, linewidth=2)
    ax.fill(theta, RCS_list, alpha=0.3)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("RCS Pattern (dB)" if use_dB else "RCS Pattern", va='bottom')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs):
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    test_cameras = scene.getTestCameras()
    os.makedirs(os.path.join(args.model_path, 'eval', 'render'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'gt'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'normal'), exist_ok=True)
    if gaussians.use_pbr:
        os.makedirs(os.path.join(args.model_path, 'eval', 'base_color'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'roughness'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'lights'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'local'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'global'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'visibility'), exist_ok=True)

    progress_bar = tqdm(range(0, len(test_cameras)), desc="Evaluating",
                        initial=0, total=len(test_cameras))

    with torch.no_grad():
        for idx in progress_bar:
            viewpoint = test_cameras[idx]
            results = render_fn(viewpoint, gaussians, pipe, background,
                                opt=opt, is_training=False,
                                dict_params=pbr_kwargs)
            if gaussians.use_pbr:
                image = results["pbr"]
            else:
                image = results["render"]

            image = torch.clamp(image, 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()

            save_image(image, os.path.join(args.model_path, 'eval', "render", f"{viewpoint.image_name}.png"))
            save_image(gt_image, os.path.join(args.model_path, 'eval', "gt", f"{viewpoint.image_name}.png"))
            save_image(results["normal"] * 0.5 + 0.5,
                       os.path.join(args.model_path, 'eval', "normal", f"{viewpoint.image_name}.png"))
            if gaussians.use_pbr:
                save_image(results["base_color"],
                           os.path.join(args.model_path, 'eval', "base_color", f"{viewpoint.image_name}.png"))
                save_image(results["roughness"],
                           os.path.join(args.model_path, 'eval', "roughness", f"{viewpoint.image_name}.png"))
                save_image(results["lights"],
                           os.path.join(args.model_path, 'eval', "lights", f"{viewpoint.image_name}.png"))
                save_image(results["local_lights"],
                           os.path.join(args.model_path, 'eval', "local", f"{viewpoint.image_name}.png"))
                save_image(results["global_lights"],
                           os.path.join(args.model_path, 'eval', "global", f"{viewpoint.image_name}.png"))
                save_image(results["visibility"],
                           os.path.join(args.model_path, 'eval', "visibility", f"{viewpoint.image_name}.png"))

    psnr_test /= len(test_cameras)
    ssim_test /= len(test_cameras)
    lpips_test /= len(test_cameras)
    with open(os.path.join(args.model_path, 'eval', "eval.txt"), "w") as f:
        f.write(f"psnr: {psnr_test}\n")
        f.write(f"ssim: {ssim_test}\n")
        f.write(f"lpips: {lpips_test}\n")
    print("\n[ITER {}] Evaluating {}: PSNR {} SSIM {} LPIPS {}".format(args.iterations, "test", psnr_test, ssim_test,
                                                                           lpips_test))


def eval_RF_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs):
    lpips_test = 0.0
    test_cameras = scene.getTestCameras()
    os.makedirs(os.path.join(args.model_path, 'eval', 'render'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'gt'), exist_ok=True)
    if gaussians.use_pbr:
        os.makedirs(os.path.join(args.model_path, 'eval', 'roughness'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'reflection_coe'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'reflection_phase'), exist_ok=True)

    progress_bar = tqdm(range(0, len(test_cameras)), desc="Evaluating",
                        initial=0, total=len(test_cameras))

    Render_RFS = []
    gt_RFS = []
    gt_indexs = []
    with torch.no_grad():
        for idx in progress_bar:
            viewpoint = test_cameras[idx]
            results = render_fn(viewpoint, gaussians, pipe, background, opt=opt, is_training=False, dict_params=pbr_kwargs)

            Render_RF = torch.abs(results["render"].cpu()).to("cuda")
            Render_RFS.append(Render_RF)

            gt_raw = viewpoint.original_image[0, :].cuda()
            gt_CSI = (gt_raw[0] + 1j * gt_raw[1])
            gt_abs = torch.sqrt(gt_CSI.real ** 2 + gt_CSI.imag ** 2 + 1e-12)
            gt_RFS.append(gt_abs)

            gt_index = viewpoint.image_name
            gt_indexs.append(gt_index)
            if gaussians.use_pbr:
                roughness_image = results["roughness"].squeeze(0).unsqueeze(0).repeat(3, 1, 1)
                reflection_coe = results["reflection_coe"].squeeze(0).unsqueeze(0).repeat(3, 1, 1)
                reflection_phase = results["reflection_phase"].squeeze(0).unsqueeze(0).repeat(3, 1, 1)
                min_rp_val = reflection_phase.min()
                max_rp_val = reflection_phase.max()
                normalized_reflection_phase = (reflection_phase - min_rp_val) / (max_rp_val - min_rp_val)

                min_ro_val = reflection_phase.min()
                max_ro_val = reflection_phase.max()
                normalized_roughness_image = (reflection_phase - min_ro_val) / (max_ro_val - min_ro_val)

                min_rc_val = reflection_coe.min()
                max_rc_val = reflection_coe.max()
                normalized_reflection_coe = (reflection_coe - min_rc_val) / (max_rc_val - min_rc_val)

                save_image(normalized_roughness_image,
                           os.path.join(args.model_path, 'eval', "roughness", f"{idx}.png"))
                save_image(normalized_reflection_coe,
                           os.path.join(args.model_path, 'eval', "reflection_coe", f"{idx}.png"))
                save_image(normalized_reflection_phase,
                           os.path.join(args.model_path, 'eval', "reflection_phase", f"{idx}.png"))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)  #
    pp = PipelineParams(parser)  #
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument('--gui', action='store_true', default=False, help="use gui")
    parser.add_argument('-t', '--type', choices=['render', 'normal', 'render_RF_bb', 'render_RF','neilf_RSSI_prediction'], default='render')
    parser.add_argument('-f', '--frequency', choices=['single', 'wide'], default='single')
    parser.add_argument("-freq", "--freq", choices=['2.4G', '5.8G', 'wideband'], default=None)
    parser.add_argument("--test_interval", type=int, default=2500)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--power_scale", type=float, default=1)
    args = parser.parse_args(sys.argv[1:])
    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    is_pbr = args.type in ['render_RF', 'neilf_RSSI_prediction']
    training(lp.extract(args), op.extract(args), pp.extract(args), is_pbr=is_pbr)

    # All done
    print("\nTraining complete.")








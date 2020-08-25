import argparse
import os
import time

import imageio
import numpy as np
import torch
import torchvision
import yaml
from tqdm import tqdm

from nerf import (
    CfgNode,
    get_ray_bundle,
    load_blender_data,
    load_llff_data,
    models,
    get_embedding_function,
    run_one_iter_of_nerf,
    mse2psnr,
)

def export_obj(vertices, triangles, diffuse, normals, filename):
    """
    Exports a mesh in the (.obj) format.
    """
    print('Writing to obj')

    with open(filename, 'w') as fh:

        for index, v in enumerate(vertices):
            fh.write("v {} {} {} {} {} {}\n".format(*v, *diffuse[index]))

        for n in normals:
            fh.write("vn {} {} {}\n".format(*n))

        for f in triangles:
            fh.write("f")
            for index in f:
                fh.write(" {}//{}".format(index + 1, index + 1))

            fh.write("\n")

    print('Finished writing to obj')


def cast_to_image(tensor, dataset_type):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Convert to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    return img
    # # Map back to shape (3, H, W), as tensorboard needs channels first.
    # return np.moveaxis(img, [-1], [0])


def cast_to_disparity_image(tensor):
    img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    img = img.clamp(0, 1) * 255
    return img.detach().cpu().numpy().astype(np.uint8)


def export_point_cloud_sample(it, ray_origins, ray_directions, depth_fine, dep_target):
    vertices_output = (ray_origins + ray_directions * depth_fine[..., None]).view(-1, 3)
    vertices_target = (ray_origins + ray_directions * dep_target[..., None]).view(-1, 3)
    vertices = torch.cat((vertices_output, vertices_target), dim = 0)
    diffuse_output = torch.zeros_like(vertices_output)
    diffuse_output[:, 0] = 1.0
    diffuse_target = torch.zeros_like(vertices_target)
    diffuse_target[:, 2] = 1.0
    diffuse = torch.cat((diffuse_output, diffuse_target), dim = 0)
    normals = torch.cat((-ray_directions.view(-1, 3), -ray_directions.view(-1, 3)), dim = 0)
    export_obj(vertices, [], diffuse, normals, f"{it:04d}.obj")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint / pre-trained model to evaluate.",
    )
    parser.add_argument(
        "--save-dir", type=str, help="Save images to this directory, if specified."
    )
    parser.add_argument(
        "--save-disparity-image", action="store_true", help="Save disparity images too."
    )
    parser.add_argument('--synthesis-images', dest = 'synthesis_images', action = 'store_true')
    parser.add_argument('--no-synthesis-images', dest = 'synthesis_images', action = 'store_false')
    parser.set_defaults(synthesis_images = False)

    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    images, poses, depth_imgs, render_poses, hwf = None, None, None, None, None
    i_train, i_val, i_test = None, None, None
    if cfg.dataset.type.lower() == "blender":
        # Load blender dataset
        images, poses, depth_imgs, render_poses, hwf, i_split = load_blender_data(
            cfg.dataset.basedir,
            categories=["test_high_res"],
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
        )

        i_test = i_split[0]
        H, W, focal = hwf
        H, W = int(H), int(W)
    elif cfg.dataset.type.lower() == "llff":
        # Load LLFF dataset
        images, poses, bds, render_poses, i_test = load_llff_data(
            cfg.dataset.basedir, factor=cfg.dataset.downsample_factor,
        )
        hwf = poses[0, :3, -1]
        H, W, focal = hwf
        hwf = [int(H), int(W), focal]
        render_poses = torch.from_numpy(render_poses)

    # Device on which to run.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
    )
    model_coarse.to(device)

    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
        )
        model_fine.to(device)

    checkpoint = torch.load(configargs.checkpoint)
    model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
    if checkpoint["model_fine_state_dict"]:
        try:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        except:
            print(
                "The checkpoint has a fine-level model, but it could "
                "not be loaded (possibly due to a mismatched config file."
            )
    if "height" in checkpoint.keys():
        hwf[0] = checkpoint["height"]
    if "width" in checkpoint.keys():
        hwf[1] = checkpoint["width"]
    if "focal_length" in checkpoint.keys():
        hwf[2] = checkpoint["focal_length"]

    model_coarse.eval()
    if model_fine:
        model_fine.eval()

    if configargs.synthesis_images:
        eval_poses = render_poses.float()
    else:
        eval_poses = poses.float()

    # Evaluation loop
    times_per_image = []
    for i, index in enumerate(tqdm(torch.randperm(i_test.shape[0]))):
        pose = eval_poses[index].to(device)
        img_target = images[index].to(device)
        dep_target = depth_imgs[index].to(device)

        start = time.time()
        rgb = None, None
        disp = None, None
        with torch.no_grad():
            pose = pose[:3, :4]
            ray_origins, ray_directions = get_ray_bundle(hwf[0], hwf[1], hwf[2], pose)
            rgb_coarse, z_vals, weights, depth_coarse, rgb_fine, disp_fine, _, depth_fine, radiance = run_one_iter_of_nerf(
                hwf[0],
                hwf[1],
                hwf[2],
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="validation",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )

            rgb = rgb_fine if rgb_fine is not None else rgb_coarse
            if configargs.save_disparity_image:
                pass
                # disp = disp_fine if disp_fine is not None else disp_coarse

        print(rgb_coarse.shape)
        coarse_loss = torch.nn.functional.mse_loss(
            rgb_coarse[..., :3], img_target[..., :3]
        )

        fine_loss = torch.tensor(0)
        if rgb_fine is not None:
            fine_loss = torch.nn.functional.mse_loss(
                rgb_fine[..., :3], img_target[..., :3]
            )

        coarse_depth_loss = torch.nn.functional.mse_loss(depth_coarse, dep_target)
        fine_depth_loss = torch.tensor(0.)
        if depth_fine is not None:
            fine_depth_loss = torch.nn.functional.mse_loss(depth_fine, dep_target)

        print(f"Loss MSE image {i}: Coarse Loss: {coarse_loss} / Fine Loss: {fine_loss}")
        print(f"Loss PSNR image {i}: Coarse PSNR: {mse2psnr(coarse_loss.item())} / Fine PSNR: {mse2psnr(fine_loss.item())}")
        print(f"Loss Depth image {i}: Coarse Depth: {coarse_depth_loss} / Fine Depth: {fine_depth_loss}")

        depth_target = depth_coarse if depth_fine is None else depth_fine
        export_point_cloud_sample(i, ray_origins, ray_directions, depth_target, dep_target)

        times_per_image.append(time.time() - start)
        if configargs.save_dir:
            savefile = os.path.join(configargs.save_dir, f"{i:04d} depth_target.png")
            imageio.imwrite(savefile, cast_to_disparity_image(dep_target))

            savefile = os.path.join(configargs.save_dir, f"{i:04d} depth_output.png")
            imageio.imwrite(savefile, cast_to_disparity_image(depth_target))

            savefile = os.path.join(configargs.save_dir, f"{i:04d}.png")
            imageio.imwrite(
                savefile, cast_to_image(rgb[..., :3], cfg.dataset.type.lower())
            )

            savefile = os.path.join(configargs.save_dir, f"{i:04d} target.png")
            imageio.imwrite(
                savefile, cast_to_image(img_target[..., :3], cfg.dataset.type.lower())
            )

            if configargs.save_disparity_image:
                savefile = os.path.join(configargs.save_dir, "disparity", f"{i:04d}.png")
                imageio.imwrite(savefile, cast_to_disparity_image(disp))

        tqdm.write(f"Avg time per image: {sum(times_per_image) / (i + 1)}")

        if (i + 1) % 5 == 0:
            print(i)
            exit(-1)


if __name__ == "__main__":
    main()

import argparse
import glob
import os
import time

import numpy as np
import torch
import torchvision
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                  load_blender_data, load_llff_data, meshgrid_xy, models,
                  mse2psnr)

# from nerf import (run_one_iter_of_nerf)

from nerf.var import (run_one_iter_of_nerf)


def main():
    torch.set_printoptions(threshold = 100, edgeitems = 50, precision = 7)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )

    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # If a pre-cached dataset is available, skip the dataloader.
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split, depth_data = None, None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    if hasattr(cfg.dataset, "cachedir") and os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data"))
        validation_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "val", "*.data")
        )
        USE_CACHED_DATASET = True
    else:
        # Load dataset
        images, poses, depth_data, render_poses, (H, W, focal), i_split = load_blender_data(
            cfg.dataset.basedir,
            categories=["test_high_res", "val"],
        )
        i_train, i_val = i_split

        H, W = int(H), int(W)
        if cfg.nerf.train.white_background:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Initialize a coarse-resolution model.
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

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    # if model_fine is not None:
    #     trainable_parameters += list(model_fine.parameters())

    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )

    optimizer_fine = getattr(torch.optim, cfg.optimizer.type)(
        list(model_fine.parameters()), lr=cfg.optimizer.lr
    )

    # Setup logging.
    log_dir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    try:
        os.makedirs(log_dir)
    except OSError:
        root_path = log_dir
        counter = 1
        while os.path.exists(log_dir):
            log_dir = root_path + f"_unnamed_{counter}"
            counter += 1

        # Create main folder
        os.makedirs(log_dir)

    # Create a writer
    print(f"Writing to {log_dir}")
    writer = SummaryWriter(log_dir)

    # Write out config parameters.
    with open(os.path.join(log_dir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # By default, start at iteration 0 (unless a checkpoint is specified).
    start_iter = 0

    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"]

    # TODO: Prepare raybatch tensor if batching random rays
    for i in trange(start_iter, cfg.experiment.train_iters):

        model_coarse.train()
        if model_fine:
            model_fine.train()

        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        depth_data_s = None
        if USE_CACHED_DATASET:
            datafile = np.random.choice(train_paths)
            cache_dict = torch.load(datafile)
            ray_bundle = cache_dict["ray_bundle"].to(device)
            ray_origins, ray_directions = (
                ray_bundle[0].reshape((-1, 3)),
                ray_bundle[1].reshape((-1, 3)),
            )
            target_ray_values = cache_dict["target"][..., :3].reshape((-1, 3))
            select_inds = np.random.choice(
                ray_origins.shape[0],
                size=(cfg.nerf.train.num_random_rays),
                replace=False,
            )
            ray_origins, ray_directions = (
                ray_origins[select_inds],
                ray_directions[select_inds],
            )
            target_ray_values = target_ray_values[select_inds].to(device)
            # ray_bundle = torch.stack([ray_origins, ray_directions], dim=0).to(device)

            rgb_coarse, _, _, depth_coarse, rgb_fine, _, _, depth_fine = run_one_iter_of_nerf(
                cache_dict["height"],
                cache_dict["width"],
                cache_dict["focal_length"],
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                it = i,
            )
        else:
            # using the test set instead of train
            img_idx = np.random.choice(i_train)
            img_target = images[img_idx].to(device)
            pose_target = poses[img_idx, :3, :4].to(device)
            depth_data_target = depth_data[img_idx].to(device)

            # generate training data
            ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)

            # generates 2x by copying the arange output across consecutive dimensions, reverse the values when stacking
            coords = torch.stack(
                meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
                dim=-1,
            )

            # create a list of H * W indices in form of (width, height), H * W * 2
            coords = coords.reshape((-1, 2))
            select_inds = np.random.choice(
                coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False
            )
            select_inds = coords[select_inds]
            ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
            ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
            # batch_rays = torch.stack([ray_origins, ray_directions], dim=0)
            target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]
            depth_data_s = depth_data_target[select_inds[:, 0], select_inds[:, 1]]

            then = time.time()
            rgb_coarse, depth_coarse, depth_std_coarse, rgb_fine, depth_fine, depth_std_fine = run_one_iter_of_nerf(
                H,
                W,
                focal,
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                depth_data = depth_data_s,
                it = i,
            )

            target_ray_values = target_s

        # rgb loss
        coarse_loss = torch.nn.functional.mse_loss(
            rgb_coarse[..., :3], target_ray_values[..., :3]
        )

        fine_loss = torch.tensor(0)
        if rgb_fine is not None:
            fine_loss = torch.nn.functional.mse_loss(
                rgb_fine[..., :3], target_ray_values[..., :3]
            )

        # depth loss
        coarse_depth_loss, fine_depth_loss = torch.tensor(0), torch.tensor(0)
        coarse_depth_empty, coarse_depth_space = torch.tensor(0), torch.tensor(0)
        mask = depth_data_s > 0
        if depth_data is not None:
            coarse_depth_loss = torch.nn.functional.mse_loss(
                depth_coarse, depth_data_s
            )

            coarse_depth_empty = torch.nn.functional.mse_loss(
                depth_coarse[~mask], depth_data_s[~mask]
            )

            coarse_depth_space = torch.nn.functional.mse_loss(
                depth_coarse[mask], depth_data_s[mask]
            )

            if depth_fine is not None:
                fine_depth_loss = torch.nn.functional.mse_loss(
                    depth_fine, depth_data_s
                )


        #loss = coarse_loss + coarse_depth_loss * 1e-2
        loss = coarse_loss + coarse_depth_loss * 1e-2
        # loss_b = fine_loss + fine_depth_loss * 1e-2
        # loss = coarse_loss + fine_loss + (coarse_depth_loss + fine_depth_loss)
        psnr = mse2psnr(coarse_loss.item())

        loss.backward()
        #loss_b.backward()
        optimizer.step()
        # optimizer_fine.step()

        optimizer.zero_grad()
        optimizer_fine.zero_grad()

        # Learning rate updates
        num_decay_steps = cfg.scheduler.lr_decay * 1000
        lr_new = cfg.optimizer.lr * (
            cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write(
                "[TRAIN] Iter: "
                + str(i)
                + " Loss: "
                + str(loss.item())
                + " Coarse: "
                + str(coarse_loss.item())
                + " CD: "
                + str(coarse_depth_loss.item())
                + " CDEmpty: "
                + str(coarse_depth_empty.item())
                + " CDSpace: "
                + str(coarse_depth_space.item())
                + " Fine: "
                + str(fine_loss.item())
                + " FD: "
                + str(fine_depth_loss.item())
                + " PSNR: "
                + str(psnr)
            )

        writer.add_scalar("train/loss", loss.item(), i)
        writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        if rgb_fine is not None:
            writer.add_scalar("train/fine_loss", fine_loss.item(), i)

        if depth_data is not None:
            writer.add_scalar("train/coarse_depth_loss", coarse_depth_loss.item(), i)
            if depth_fine is not None:
                writer.add_scalar("train/fine_depth_loss", fine_depth_loss.item(), i)

        writer.add_scalar("train/psnr", psnr, i)

        # Validation
        if i >= 50000:
            tqdm.write("  [VAL] =======> Iter: " + str(i))
            model_coarse.eval()
            if model_fine:
                model_coarse.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                if USE_CACHED_DATASET:
                    datafile = np.random.choice(validation_paths)
                    cache_dict = torch.load(datafile)
                    rgb_coarse, _, rgb_fine, _ = run_one_iter_of_nerf(
                        cache_dict["height"],
                        cache_dict["width"],
                        cache_dict["focal_length"],
                        model_coarse,
                        model_fine,
                        cache_dict["ray_origins"].to(device),
                        cache_dict["ray_directions"].to(device),
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = cache_dict["target"].to(device)
                else:
                    img_idx = np.random.choice(i_val)
                    img_target = images[img_idx].to(device)
                    pose_target = poses[img_idx, :3, :4].to(device)
                    ray_origins, ray_directions = get_ray_bundle(
                        H, W, focal, pose_target
                    )

                    rgb_coarse, _, _, rgb_fine, _, _, = run_one_iter_of_nerf(
                        H,
                        W,
                        focal,
                        model_coarse,
                        model_fine,
                        ray_origins,
                        ray_directions,
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = img_target

                coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                loss, fine_loss = 0.0, 0.0
                if rgb_fine is not None:
                    fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                    loss = fine_loss
                else:
                    loss = coarse_loss

                loss = coarse_loss
                psnr = mse2psnr(coarse_loss.item())
                writer.add_scalar("validation/loss", coarse_loss.item(), i)
                writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                writer.add_scalar("validation/psnr", psnr, i)
                writer.add_image(
                    "validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i
                )
                if rgb_fine is not None:
                    writer.add_image(
                        "validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i
                    )
                    writer.add_scalar("validation/fine_loss", fine_loss.item(), i)

                writer.add_image(
                    "validation/img_target",
                    cast_to_image(target_ray_values[..., :3]),
                    i,
                )

                tqdm.write(
                    "  Validation loss: "
                    + str(loss.item())
                    + " Validation PSNR: "
                    + str(psnr)
                    + " Time: "
                    + str(time.time() - start)
                )

            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": None
                if not model_fine
                else model_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "psnr": psnr,
            }
            torch.save(
                checkpoint_dict,
                os.path.join(log_dir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")

            exit(-1)

    print("Done!")


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img


if __name__ == "__main__":
    print(os.getcwd())
    main()

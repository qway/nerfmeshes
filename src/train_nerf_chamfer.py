import argparse
import glob
import os
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml

from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from mesh_nerf_chamfer import export_obj, generate
from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                  load_blender_data, load_llff_data, meshgrid_xy, models,
                  mse2psnr, run_one_iter_of_nerf)




def main():
    logging.basicConfig(filename='results/result.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.DEBUG)
    val_diffuse_loss = []
    val_chamfer_loss = []
    psnr_history = []

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

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # If a pre-cached dataset is available, skip the dataloader.
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None



    if hasattr(cfg.dataset, "cachedir") and os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data"))
        validation_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "val", "*.data")
        )
        USE_CACHED_DATASET = True
    else:
        # Load dataset
        images, poses, render_poses, hwf = None, None, None, None
        if cfg.dataset.type.lower() == "blender":
            images, poses, render_poses, hwf, i_split = load_blender_data(
                cfg.dataset.basedir,
                half_res=cfg.dataset.half_res,
                testskip=cfg.dataset.testskip,
            )
            i_train, i_val, i_test = i_split
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]
            if cfg.nerf.train.white_background:
                images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        elif cfg.dataset.type.lower() == "llff":
            images, poses, bds, render_poses, i_test = load_llff_data(
                cfg.dataset.basedir, factor=cfg.dataset.downsample_factor
            )
            hwf = poses[0, :3, -1]
            poses = poses[:, :3, :4]
            if not isinstance(i_test, list):
                i_test = [i_test]
            if cfg.dataset.llffhold > 0:
                i_test = np.arange(images.shape[0])[:: cfg.dataset.llffhold]
            i_val = i_test
            i_train = np.array(
                [
                    i
                    for i in np.arange(images.shape[0])
                    if (i not in i_test and i not in i_val)
                ]
            )
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]
            images = torch.from_numpy(images)
            poses = torch.from_numpy(poses)

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # device = 'cpu'

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
    if model_fine is not None:
        trainable_parameters += list(model_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )

    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    meshdir = os.path.join(cfg.experiment.meshdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    # Write out config parameters.
    with open(os.path.join(logdir, "config.yml"), "w") as f:
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

    # # TODO: Prepare raybatch tensor if batching random rays
    ##############################################################################################

    # trg_obj = os.path.join('../data/output/lego.obj')
    trg_obj = os.path.join('meshes/lego_rotate.obj')
    # We read the target 3D model using load_obj
    verts, faces, aux = load_obj(trg_obj)

    # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
    # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
    # For this tutorial, normals and textures are ignored.
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
    # (scale, center) will be used to bring the predicted mesh to its original center and scale
    # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    # We construct a Meshes structure for the target mesh
    trg_mesh = Meshes(verts=[verts], faces=[faces_idx])
    #####################################################################################################
    for i in trange(start_iter, cfg.experiment.train_iters):

        model_coarse.train()
        if model_fine:
            model_fine.train()

        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
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

            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
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
            )
        else:
            img_idx = np.random.choice(i_train)
            img_target = images[img_idx].to(device)
            pose_target = poses[img_idx, :3, :4].to(device)
            ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
            coords = torch.stack(
                meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
                dim=-1,
            )
            coords = coords.reshape((-1, 2))
            select_inds = np.random.choice(
                coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False
            )
            select_inds = coords[select_inds]
            ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
            ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
            # batch_rays = torch.stack([ray_origins, ray_directions], dim=0)
            target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]

            then = time.time()
            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
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
            )
            target_ray_values = target_s

        coarse_loss = torch.nn.functional.mse_loss(
            rgb_coarse[..., :3], target_ray_values[..., :3]
        )
        fine_loss = None
        fine_chamfer_loss = None
        if rgb_fine is not None:
            fine_loss = torch.nn.functional.mse_loss(
                rgb_fine[..., :3], target_ray_values[..., :3]
            )
        # loss = torch.nn.functional.mse_loss(rgb_pred[..., :3], target_s[..., :3])
        loss = 0.0
        # if fine_loss is not None:
        #     loss = fine_loss
        # else:
        #     loss = coarse_loss

        loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
        # loss =  10 * original_loss + ( fine_chamfer_loss if fine_chamfer_loss is not None else 0.0)
        loss.backward()
        psnr = mse2psnr(loss.item())
        optimizer.step()
        optimizer.zero_grad()

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
                + " Diffuse Loss: "
                + str(loss.item())
                + " PSNR: "
                + str(psnr)
                # + "diffuse Loss: "
                # + str(original_loss.item())
                # + "chamfer Loss: "
                # + str(fine_chamfer_loss)
            )
            logging.info(f"[Train] Iter: {str(i)}   Diffuse Loss: {loss.item()}   PSNR: {str(psnr)}")

        writer.add_scalar("train/loss", loss.item(), i)
        writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        if rgb_fine is not None:
            writer.add_scalar("train/fine_loss", fine_loss.item(), i)
        writer.add_scalar("train/psnr", psnr, i)

        # Validation
        if (
            i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1
        ):
            tqdm.write("[VAL] =======> Iter: " + str(i))
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
                    rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
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
                    rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
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
                loss, fine_diffuse_loss, fine_chamfer_loss = 0.0, 0.0, 0.0

                ##########################################chamfer loss################################################################################
                if i >= 2 * cfg.experiment.validate_every:

                    i_check = i - cfg.experiment.validate_every
                    # checkpoint_path =
                    if os.path.exists(os.path.join(logdir, "checkpoint" + str(i_check).zfill(5) + ".ckpt")):
                        checkpoint = os.path.join(logdir, "checkpoint" + str(i_check).zfill(5) + ".ckpt")
                        # pred_obj = generate(configargs.config, checkpoint, meshdir)
                        verts, faces = generate(configargs.config, checkpoint, meshdir)
                if i >= cfg.experiment.save_every+1:
                    # We read the target 3D model using load_obj
                    # verts, faces, aux = load_obj(pred_obj)
                    verts, faces = generate(configargs.config, checkpoint, meshdir)
                    # print(verts)
                    # print(faces)
                    # verts, faces = torch.from_numpy(verts), torch.from_numpy(faces)
                    # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
                    # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
                    # For this tutorial, normals and textures are ignored.
                    # faces_idx = faces.verts_idx.to(device)
                    faces_idx = faces.to(device)
                    verts = verts.to(device)

                    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
                    # (scale, center) will be used to bring the predicted mesh to its original center and scale
                    # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
                    center = verts.mean(0)
                    verts = verts - center
                    scale = max(verts.abs().max(0)[0])
                    verts = verts / scale

                    # We construct a Meshes structure for the target mesh
                    pred_mesh = Meshes(verts=[verts], faces=[faces_idx])
                    trg_mesh = trg_mesh.to(device)
                    # https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh
                    sample_trg = sample_points_from_meshes(trg_mesh, 1600)
                    sample_pred = sample_points_from_meshes(pred_mesh, 1600)

                    fine_chamfer_loss, _ = chamfer_distance(sample_trg, sample_pred)


                if rgb_fine is not None:
                    fine_diffuse_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])

                    fine_loss = fine_diffuse_loss
                else:
                    loss = coarse_loss


                loss = coarse_loss + fine_loss
                psnr = mse2psnr(loss.item())
                writer.add_scalar("validation/loss", loss.item(), i)
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
                    "Validation loss: "
                    + str(loss.item())
                    + "Validation diffuse loss: "
                    + str(fine_diffuse_loss)
                    + "Validation chamfer loss: "
                    + str(fine_chamfer_loss)
                    + " Validation PSNR: "
                    + str(psnr)
                    + " Time: "
                    + str(time.time() - start)
                )
                logging.info(f"[Val] Iter: {str(i)}   Diffuse Loss: {loss.item()}   PSNR: {str(psnr)}     chamfer loss: {fine_chamfer_loss}")
                val_chamfer_loss.append(fine_chamfer_loss)
                val_diffuse_loss.append(loss)
                psnr_history.append(psnr)

        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": None
                if not model_fine
                else model_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "psnr": psnr,
                # "diffuse Loss": original_loss,
                "chamfer Loss": fine_chamfer_loss,
                "val_chamfer_loss": val_chamfer_loss,
                "val_diffuse_loss": val_diffuse_loss
            }
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
                # os.path.join(logdir, "checkpoint.ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")
    plt.plot(val_diffuse_loss)
    plt.plot(val_chamfer_loss)
    # plt.plot(psnr_history)
    plt.title("loss and psnr")
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend(['diffuse', 'chamfer'])
    plt.savefig('results/loss.png')

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

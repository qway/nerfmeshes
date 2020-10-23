import torch
import matplotlib.pyplot as plt

from nerf.nerf_helpers import comp_depth, get_point_clouds


class LoggerDepthProjection:
    def __init__(self, step_size, name):
        super(LoggerDepthProjection, self).__init__()
        self.step_size = step_size
        self.name = name

        # Config for Three.js points rendering
        self.config = {
            'material': {
                'cls': 'PointsMaterial',
                'size': 0.03
            }
        }

    def tick(self, logger, step, ray_origins, ray_directions, output_depth, target_depth):
        if step % self.step_size == 0 and step > 0:
            vertices, colors, _ = get_point_clouds(ray_origins, ray_directions, output_depth, target_depth)

            # Logger step
            global_step = step // self.step_size

            vertices = vertices.unsqueeze(0)
            colors = colors.unsqueeze(0)

            logger.add_mesh(self.name, vertices = vertices, colors = colors, global_step = global_step, config_dict = self.config)


class LoggerTreeWeights:
    def __init__(self, tree, name):
        super(LoggerTreeWeights, self).__init__()
        self.tree = tree
        self.name = name

        # Logger step
        self.counter = 0

    def tick(self, logger, step):
        if self.tree.ticked(step):
            y = self.tree.memm.sort().values.cpu().detach().numpy()
            x = torch.arange(0, y.shape[0]).cpu().detach().numpy()

            # Plot figure
            fig = plt.figure(figsize = (15, 10))
            plt.plot(x, y)

            logger.add_figure(self.name, fig, global_step = self.counter, close = True)

            self.counter += 1


class LoggerTree:
    def __init__(self, tree, name):
        super(LoggerTree, self).__init__()
        self.tree = tree
        self.name = name
        # Logger step
        self.counter = 0

    def tick(self, logger, step):
        if self.tree.ticked(step):
            vertices, faces, colors = self.tree.flatten()
            vertices, faces, colors = vertices.unsqueeze(0), faces.unsqueeze(0), colors.unsqueeze(0)

            logger.add_mesh(self.name, vertices=vertices, colors=colors, faces=faces, global_step = self.counter)

            self.counter += 1


class LoggerDepthLoss:
    def __init__(self, type = 'train', empty_value = 0.):
        super(LoggerDepthLoss, self).__init__()
        self.type = type
        self.empty_value = empty_value

    def tick(self, logs, out_rgb, target_rgb, out_depth, target_depth = None):
        # Depth consideration
        if target_depth is None:
            return logs

        mask = target_depth != self.empty_value

        # Diffuse loss with the respect to the depth
        rgb_surface_loss = torch.nn.functional.mse_loss(
            out_rgb[mask], target_rgb[mask]
        )

        rgb_void_loss = torch.nn.functional.mse_loss(
            out_rgb[~mask], target_rgb[~mask]
        )

        # Depth loss
        depth_loss, depth_empty, depth_space, l1 = comp_depth(out_depth, target_depth, self.empty_value)

        return {
            **logs,
            f"{self.type}/rgb_surface_loss": rgb_surface_loss,
            f"{self.type}/rgb_void_loss": rgb_void_loss,
            f"{self.type}/depth_loss": depth_loss,
            f"{self.type}/depth_surface_loss": depth_space,
            f"{self.type}/depth_void_loss": depth_empty,
            f"{self.type}/depth_l1_loss": l1,
        }

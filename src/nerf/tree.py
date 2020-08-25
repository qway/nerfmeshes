import torch
import math
import time
from typing import Optional

import torch


class Node:
    def __init__(self, config, bounds, depth):
        self.config = config
        self.bounds = bounds
        self.depth = depth
        self.max_depth = self.config.tree.max_depth
        if self.depth == 0:
            self.count = self.config.tree.subdivision_outer_count
        else:
            self.count = self.config.tree.subdivision_inner_count

        self.weight = 0.
        self.sparse = True
        self.children = []

    def subdivide(self):
        if self.depth >= self.max_depth:
            return

        offset = self.bounds[1] - self.bounds[0]
        for i in range(0, self.count):
            for g in range(0, self.count):
                for h in range(0, self.count):
                    ind1 = torch.tensor([i, g, h], dtype = torch.float) / self.count * offset
                    ind2 = torch.tensor([i + 1, g + 1, h + 1], dtype = torch.float) / self.count * offset

                    bounds = self.bounds[0] + ind1, self.bounds[0] + ind2
                    child = Node(self.config, bounds, self.depth + 1)

                    self.children.append(child)

    def clear(self):
        self.children = []


class TreeSampling:
    vertex_indices = [
        [],
        [0],
        [1],
        [2],
        [0, 1],
        [1, 2],
        [0, 2],
        [0, 1, 2],
    ]

    faces_indices = [
        0, 2, 1, 2, 4, 1,
        0, 3, 2, 2, 3, 5,
        0, 1, 6, 6, 3, 0,
        1, 4, 7, 7, 6, 1,
        3, 6, 7, 7, 5, 3,
        2, 7, 4, 7, 2, 5
    ]

    colors_tensor = torch.as_tensor([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [128, 64, 64],
        [64, 64, 128],
    ], dtype = torch.int).unsqueeze(0)

    def __init__(self, config, device):
        self.config = config
        self.device = device

        # initial bounds
        near, far = self.config.dataset.near, self.config.dataset.far
        mean = (near + far) / 2
        bounds = torch.tensor([near - mean] * 3), torch.tensor([far - mean] * 3)

        # tree root
        self.root = Node(self.config, bounds, 0)
        self.root.subdivide()

        # tree residual data
        self.voxels = None
        self.memm = None
        self.counter = 1

        # intialize
        self.consolidate()

    def flatten(self):
        vertices = []
        faces = []
        colors = []
        for node in self.root.children:
            offset = node.bounds[1] - node.bounds[0]
            offset_index = len(vertices)

            for t in range(8):
                tt = node.bounds[0].clone()
                tt[TreeSampling.vertex_indices[t]] += offset[TreeSampling.vertex_indices[t]]

                vertices.append(tt)

            colors.append(TreeSampling.colors_tensor)
            faces.append(torch.tensor(TreeSampling.faces_indices) + offset_index)

        vertices = torch.stack(vertices, 0)
        faces = torch.stack(faces, 0).view(-1, 3).int()
        colors = torch.stack(colors, 0).view(-1, 3)

        return vertices, faces, colors

    def consolidate(self, split = False):
        if self.memm is not None:
            print(f"Min mem {self.memm.min()}")
            print(f"Max mem {self.memm.max()}")

            voxels_indices = torch.arange(self.memm.shape[0])
            mask_voxels = self.memm > self.config.tree.eps
            print(f"Filtered mem {(~mask_voxels).sum()}")
            mask_voxels_list = voxels_indices[mask_voxels].tolist()
            print(f"Current voxels count {len(mask_voxels_list)}")

            children = []
            for index in mask_voxels_list:
                child = self.root.children[index]

                if split:
                    child.subdivide()
                    if len(child.children) > 0:
                        children += child.children
                    else:
                        children.append(child)
                else:
                    children.append(child)

            self.root.children = children

        self.voxels = [ torch.stack(node.bounds, 0) for node in self.root.children ]
        self.voxels = torch.stack(self.voxels, 0).to(self.device)
        self.memm = torch.zeros(self.voxels.shape[0], ).to(self.device)
        self.counter = 1

    def ray_batch_integration(self, ray_voxel_indices, ray_batch_weights):
        """ Performs ray batch integration into the nodes by weight accumulation
        Args:
            ray_voxel_indices (torch.Tensor): Tensor (RxN) batch ray voxel indices.
            ray_batch_weights (torch.Tensor): Tensor (RxN) batch ray sample weights.
        """
        voxel_count = self.voxels.shape[0]
        ray_count, ray_samples_count = ray_batch_weights.shape

        # accumulate weights
        acc = torch.zeros(ray_count, voxel_count, device = self.device)
        acc = acc.scatter_add(1, ray_voxel_indices, ray_batch_weights)

        # distribute weights (voxel/accumulations) while being numerically stable
        memm_sample = acc.sum(0)
        self.memm = self.memm + (memm_sample - self.memm) / self.counter
        self.memm = torch.clamp(self.memm, -1e12, 1e12)
        self.counter += 1

    def extract_(self, bounds, signs):
        out = bounds[signs]
        out = out.transpose(1, 2)
        out = torch.stack((out[:, :, 0, 0], out[:, :, 1, 1], out[:, :, 2, 2]), -1)

        return out[:, :, None, :]

    def batch_ray_voxel_intersect(self, bounds, origins, dirs, samples = 64):
        """ Returns batch of min and max intersections with their indices.
        Args:
            bounds (torch.Tensor): Tensor (Nx2x3) whose elements define the min/max bounds.
            origins (torch.Tensor): Tensor (Rx3) whose elements define the ray origin positions.
            dirs (torch.Tensor): Tensor (Rx3) whose elements define the ray directions.

        Returns:
            intersections (torch.Tensor): min/max intersections ray direction scalars
            indices (torch.Tensor): indices of valid intersections
        """
        assert origins.shape[0] == dirs.shape[0], "Batch ray size not consistent"
        rays_count, voxels_count = origins.shape[0], bounds.shape[0],

        inv_dirs = 1 / dirs
        signs = (inv_dirs < 0).long()
        inv_signs = 1 - signs
        origins = origins[:, None, None, :]
        inv_dirs = inv_dirs[:, None, None, :]
        bounds = bounds.transpose(0, 1)

        tvmin = ((self.extract_(bounds, signs) - origins) * inv_dirs).squeeze(2)
        tvmax = ((self.extract_(bounds, inv_signs) - origins) * inv_dirs).squeeze(2)

        mask = torch.ones((rays_count, voxels_count,), dtype = torch.bool, device = bounds.device)

        # y-axis filter & intersection
        # DeMorgan's law ~(tvmin[..., 0] > tvmax[..., 1] or tvmin[..., 1] > tvmax[..., 0])]
        mask = mask & (tvmin[..., 0] <= tvmax[..., 1]) & (tvmin[..., 1] <= tvmax[..., 0])

        # y-axis
        mask_miny = tvmin[..., 1] > tvmin[..., 0]
        tvmin[..., 0][mask_miny] = tvmin[mask_miny][..., 1]

        mask_maxy = tvmax[..., 1] < tvmax[..., 0]
        tvmax[..., 0][mask_maxy] = tvmax[mask_maxy][..., 1]

        # z-axis filter & intersection
        # DeMorgan's law ~(tvmin[..., 0] > tvmax[..., 2]) or (tvmin[..., 2] > tvmax[..., 0])
        mask = mask & (tvmin[..., 0] <= tvmax[..., 2]) & (tvmin[..., 2] <= tvmax[..., 0])

        # z-axis
        mask_minz = tvmin[..., 2] > tvmin[..., 0]
        tvmin[..., 0][mask_minz] = tvmin[mask_minz][..., 2]

        mask_maxz = tvmax[..., 2] < tvmax[..., 0]
        tvmax[..., 0][mask_maxz] = tvmax[mask_maxz][..., 2]

        # find intersection scalars
        intersections = torch.stack((tvmin[..., 0], tvmax[..., 0]), -1)
        intersections[~mask] = 0

        # mask outliers
        ray_mask = (intersections.sum(-1) != 0).sum(-1) > 0
        intersections = intersections[ray_mask]

        weights = torch.ones((rays_count, voxels_count,), device = bounds.device)
        weights[~mask] = 1e-9

        samples = torch.multinomial(weights[ray_mask], samples, replacement = True)
        samples_indices = samples[..., None].expand(-1, -1, 2)
        values = intersections.gather(-2, samples_indices)

        ray_rel = ray_mask.sum()
        indices = torch.arange(voxels_count, device = bounds.device).repeat(ray_rel, 1).gather(-1, samples)

        z_vals = values[..., 0] + (values[..., 1] - values[..., 0]) * torch.rand_like(values[..., 0],                                                                                      device = bounds.device)
        z_vals, indices_ordered = z_vals.sort(-1)

        indices = indices.gather(-1, indices_ordered)

        return z_vals, indices, intersections, ray_mask


def create_scene(data):
    vertices = []
    faces = []
    colors = []

    acc = 0
    for component in data:
        vertices.append(component[0])
        faces.append(component[1] + acc)
        colors.append(component[2])

        acc += component[0].shape[0]

    vertices = torch.cat(vertices, 0)
    faces = torch.cat(faces, 0)
    colors = torch.cat(colors, 0)

    return vertices, faces, colors
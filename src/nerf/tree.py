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
        [0, 0, 0],
        [128, 128, 128],
        [128, 128, 128],
        [128, 128, 128],
        [0, 0, 0],
        [128, 128, 128],
        [0, 0, 0],
        [128, 128, 128],
    ], dtype=torch.int).unsqueeze(0)

    def __init__(self, config, device):
        self.config = config
        self.device = device

        # Initial bounds, normalized
        self.ray_near, self.ray_far = self.config.dataset.near, self.config.dataset.far
        self.ray_mean = (self.ray_near + self.ray_far) / 2
        bounds = torch.tensor([self.ray_near - self.ray_mean] * 3), torch.tensor([self.ray_far - self.ray_mean] * 3)

        # Tree root
        self.root = Node(self.config, bounds, 0)
        self.root.subdivide()

        # Tensor (Nx2x3) whose elements define the min/max bounds.
        self.voxels = None

        # Tree residual data
        self.memm = None
        self.counter = 1

        # Initialize
        self.consolidate()

    def ticked(self, step):
        tree_config = self.config.tree
        step_size_tree = tree_config.step_size_tree
        step_size_integration_offset = tree_config.step_size_integration_offset
        if step > step_size_integration_offset:
            curr_step = step - step_size_integration_offset
            return curr_step > 0 and curr_step % step_size_tree == 0

        return False

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
            print(f"Min memm {self.memm.min()}")
            print(f"Max memm {self.memm.max()}")
            print(f"Mean memm {self.memm.mean()}")
            print(f"Median memm {self.memm.median()}")
            print(f"Threshold {self.config.tree.eps}")

            # Filtering
            voxels_indices = torch.arange(self.memm.shape[0])
            mask_voxels = self.memm > self.config.tree.eps
            mask_voxels_list = voxels_indices[mask_voxels].tolist()
            inv_weights = (1.0 - self.memm[mask_voxels]).tolist()

            voxel_count_initial = voxels_indices.shape[0]
            voxel_count_filtered = (~mask_voxels).sum()
            voxel_count_current = len(mask_voxels_list)
            print(f"From {voxel_count_initial} voxels with {voxel_count_filtered} filtered to current {voxel_count_current}")

            # Nodes closer to the root with high weight have higher priority
            voxels_filtered = [ self.root.children[index] for index in mask_voxels_list ]
            voxels_filtered = sorted(enumerate(voxels_filtered), key = lambda item: (item[1].depth, inv_weights[item[0]]))
            voxels_filtered = [ item[1] for item in voxels_filtered ]

            inner_size = self.config.tree.subdivision_inner_count ** 3 - 1

            children = []
            for index, child in enumerate(voxels_filtered):
                # Check if exceeds max cap
                exp_voxel_count = len(children) + inner_size + voxel_count_current - index
                if exp_voxel_count < self.config.tree.max_voxel_count:
                    child.subdivide()
                    if len(child.children) > 0:
                        children += child.children
                    else:
                        children.append(child)
                else:
                    children.append(child)

            print(f"Now {len(children)} voxels")
            self.root.children = children

        self.voxels = [ torch.stack(node.bounds, 0) for node in self.root.children ]
        if len(self.voxels) == 0:
            print(f"The chosen threshold {self.config.tree.eps} was set too high!")

        self.voxels = torch.stack(self.voxels, 0).to(self.device)
        self.memm = torch.zeros(self.voxels.shape[0], ).to(self.device)
        self.counter = 1

    def ray_batch_integration(self, step, ray_voxel_indices, ray_batch_weights, ray_batch_weights_mask):
        """ Performs ray batch integration into the nodes by weight accumulation
        Args:
            step (int): Training step.
            ray_voxel_indices (torch.Tensor): Tensor (RxN) batch ray voxel indices.
            ray_batch_weights (torch.Tensor): Tensor (RxN) batch ray sample weights.
            ray_batch_weights_mask (torch.Tensor): Tensor (RxN) batch ray sample weights mask.
        """
        if step < self.config.tree.step_size_integration_offset:
            return
        elif step == self.config.tree.step_size_integration_offset:
            print(f"Began ray batch integration... Step:{step}")

        voxel_count = self.voxels.shape[0]
        ray_count, ray_samples_count = ray_batch_weights.shape

        # accumulate weights
        acc = torch.zeros(ray_count, voxel_count, device = self.device)
        acc = acc.scatter_add(-1, ray_voxel_indices, ray_batch_weights)
        acc = acc.sum(0)

        # freq weights
        freq = torch.zeros(ray_count, voxel_count, device = self.device)
        freq = freq.scatter_add(-1, ray_voxel_indices, ray_batch_weights_mask)
        freq = freq.sum(0)
        mask = freq > 0

        # distribute weights (voxel/accumulations) while being numerically stable
        self.memm[mask] += (acc[mask] / freq[mask] - self.memm[mask]) / self.counter
        self.counter += 1

    def extract_(self, bounds, signs):
        out = bounds[signs]
        out = out.transpose(1, 2)
        out = out[:, :, [0, 1, 2], [0, 1, 2]]

        return out[:, :, None, :]

    def uniform_sampling_(self, tensor, count):
        indices = torch.arange(0, tensor.shape[-1], device = tensor.device).expand(tensor.shape)
        samples_count = tensor.long().sum(-1)
        output = tensor.long() * (count // samples_count)[:, None]
        remainder = count - output.sum(-1)

        rem1 = torch.stack((remainder, tensor.sum(-1) - remainder), -1).flatten()
        rem2 = torch.stack((torch.ones_like(remainder), torch.zeros_like(remainder)), -1).flatten()
        remaining = rem2.repeat_interleave(rem1, 0)

        output[tensor > 0] += remaining
        samples = indices[tensor].repeat_interleave(output[tensor], -1).view(-1, count)

        return samples

    def batch_ray_voxel_intersect(self, origins, dirs, samples_count = 64):
        """ Returns batch of min and max intersections with their indices.
        Args:
            origins (torch.Tensor): Tensor (1x3) whose elements define the ray origin positions.
            dirs (torch.Tensor): Tensor (Rx3) whose elements define the ray directions.

        Returns:
            intersections (torch.Tensor): min/max intersections ray direction scalars
            indices (torch.Tensor): indices of valid intersections
        """
        bounds = self.voxels
        rays_count, voxels_count = dirs.shape[0], bounds.shape[0],

        inv_dirs = 1 / dirs
        signs = (inv_dirs < 0).long()
        inv_signs = 1 - signs
        origins = origins[:, None, None, :]
        inv_dirs = inv_dirs[:, None, None, :]
        bounds = bounds.transpose(0, 1)

        # Min, max intersections
        tvmin = ((self.extract_(bounds, signs) - origins) * inv_dirs).squeeze(2)
        tvmax = ((self.extract_(bounds, inv_signs) - origins) * inv_dirs).squeeze(2)

        # Keep track non-intersections
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

        # find intersection scalars within range [ near, far ]
        intersections = torch.stack((tvmin[..., 0], tvmax[..., 0]), -1)

        # mask outliers
        ray_mask = mask.sum(-1) > 0

        # see this https://github.com/pytorch/pytorch/issues/43768
        ray_rel = ray_mask.sum()
        if ray_rel == 0:
            indices = torch.ones(0, samples_count, device = bounds.device)

            return torch.rand_like(indices), indices.long(), ray_mask

        if self.config.tree.use_voxel_random_sampling:
            # apply small weight for non-intersections
            weights = torch.ones((rays_count, voxels_count,), device = bounds.device)

            # apply noise
            weights[~mask] = 1e-12

            # sample intersections
            samples = torch.multinomial(weights, samples_count, replacement = True)
        else:
            # uniform interleaved deterministic sampling
            samples = self.uniform_sampling_(mask, samples_count)

        # Gather intersection samples
        samples_indices = samples[..., None].expand(-1, -1, 2)
        values = intersections.gather(-2, samples_indices)

        values_min, values_max = values[..., 0], values[..., 1]
        if self.config.tree.use_depth_random_sampling:
            # Random sampling
            value_samples = torch.rand_like(values_min, device = bounds.device)
            z_vals = values_min + (values_max - values_min) * value_samples
        else:
            # Deterministic sampling
            minxx = values_min.min(-1).values
            maxxx = values_max.max(-1).values

            value_samples = torch.linspace(0., 1., samples_count, device = bounds.device).expand(values_min.shape)
            z_vals = minxx[:, None] + (maxxx - minxx)[:, None] * value_samples

        z_vals, indices_ordered = z_vals.sort(-1)
        indices = samples.gather(-1, indices_ordered)

        return z_vals, indices, ray_mask

    def serialize(self):
        return {
            "root": self.root,
            "voxels": self.voxels,
            "memm": self.memm,
            "counter": self.counter
        }

    def deserialize(self, dict):
        print("Loaded tree from checkpoint...")
        self.root = dict["root"]
        self.voxels = dict["voxels"].to(self.device)
        self.memm = dict["memm"].to(self.device)
        self.counter = dict["counter"]


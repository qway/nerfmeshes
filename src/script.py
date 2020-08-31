import torch

def extract_(bounds, signs):
    out = bounds[signs]
    out = out.transpose(1, 2)
    out = out[:, :, [0, 1, 2], [0, 1, 2]]

    return out[:, :, None, :]


def batch_ray_voxel_intersect(bounds, origins, dirs, samples = 64, verbose = False):
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

    tvmin = ((extract_(bounds, signs) - origins) * inv_dirs).squeeze(2)
    tvmax = ((extract_(bounds, inv_signs) - origins) * inv_dirs).squeeze(2)

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
    # print(intersections.gather(-2, torch.ones((0, 64, 2), device = device).long()))
    print(intersections.gather(-2, torch.ones((0, 64, 2), device = device).long()))

    weights = torch.ones((rays_count, voxels_count,), device = bounds.device)
    weights[~mask] = 1e-9

    samples = torch.multinomial(weights[ray_mask], samples, replacement = True)
    print(samples.shape)
    samples_indices = samples[..., None].expand(-1, -1, 2)

    # print(samples_indices.shape)
    print(intersections.gather(-2, torch.ones((0, 64, 2), device = device).long()))

    values = intersections.gather(-2, samples_indices)

    ray_rel = ray_mask.sum()
    indices = torch.arange(voxels_count, device = bounds.device).repeat(ray_rel, 1).gather(-1, samples)

    z_vals = values[..., 0] + (values[..., 1] - values[..., 0]) * torch.rand_like(values[..., 0],
                                                                                  device = bounds.device)
    z_vals, indices_ordered = z_vals.sort(-1)

    indices = indices.gather(-1, indices_ordered)

    return z_vals, indices, intersections, ray_mask

if __name__ == "__main__":
    sample = torch.load("sample_rec")

    print(torch.__version__)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tree, ro, rd = sample['tree'], sample['ro'], sample['rd']

    tree = tree.to(device)
    ro = ro.to(device)
    rd = rd.to(device)

    z_vals, indices, intersections, ray_mask = batch_ray_voxel_intersect(tree, ro, rd, verbose = True)

    print(device)
    print(z_vals.shape)
import warnings
import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as Rot
import os, json
import torch
import random
import cubvh
import matplotlib.pyplot as plt

from packaging import version as pver

import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

W = 800
H = 800

# ref: https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367
def chamfer_distance(x, y, metric='l2', direction='bi'):
    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = (np.mean(min_y_to_x) + np.mean(min_x_to_y)) / 2 # modified to keep scale
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, patch_size=1, coords=None):
    ''' get rays
    Args:
        poses: [N/1, 4, 4], cam2world
        intrinsics: [N/1, 4] tensor or [4] ndarray
        H, W, N: int
    Returns:
        rays_o, rays_d: [N, 3]
        i, j: [N]
    '''

    device = poses.device
    
    if isinstance(intrinsics, np.ndarray):
        fx, fy, cx, cy = intrinsics
    else:
        fx, fy, cx, cy = intrinsics[:, 0], intrinsics[:, 1], intrinsics[:, 2], intrinsics[:, 3]

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5

    results = {}

    if N > 0:
       
        if coords is not None:
            inds = coords[:, 0] * W + coords[:, 1]

        elif patch_size > 1:

            # random sample left-top cores.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten


        else: # random sampling
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['i'] = i.long()
        results['j'] = j.long()

    else:
        inds = torch.arange(H*W, device=device)

    zs = -torch.ones_like(i) # z is flipped
    xs = (i - cx) / fx
    ys = -(j - cy) / fy # y is flipped
    directions = torch.stack((xs, ys, zs), dim=-1) # [N, 3]
    # do not normalize to get actual depth, ref: https://github.com/dunbar12138/DSNeRF/issues/29
    # directions = directions / torch.norm(directions, dim=-1, keepdim=True) 
    rays_d = (directions.unsqueeze(1) @ poses[:, :3, :3].transpose(-1, -2)).squeeze(1) # [N, 1, 3] @ [N, 3, 3] --> [N, 1, 3]

    rays_o = poses[:, :3, 3].expand_as(rays_d) # [N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    # visualize_rays(rays_o[0].detach().cpu().numpy(), rays_d[0].detach().cpu().numpy())

    return results


def sample_surface(poses, intrinsics, mesh, N):

    # normalize
    vmin, vmax = mesh.bounds
    center = (vmin + vmax) / 2
    scale = 1 / (vmax - vmin)
    mesh.vertices = (mesh.vertices - center) * scale

    RT = cubvh.cuBVH(mesh.vertices, mesh.faces)

    # need to cast rays ...
    all_positions = []

    per_frame_n = N // len(poses)

    for pose in poses:
        
        pose = torch.from_numpy(pose).unsqueeze(0).cuda()
        rays = get_rays(pose, intrinsics, H, W, -1)
        rays_o = rays['rays_o'].contiguous().view(-1, 3)
        rays_d = rays['rays_d'].contiguous().view(-1, 3)

        positions, face_id, depth = RT.ray_trace(rays_o, rays_d)

        # depth = depth.detach().cpu().numpy().reshape(H, W, 1)
        # mask = depth >= 10
        # mn = depth[~mask].min()
        # mx = depth[~mask].max()
        # depth = (depth - mn) / (mx - mn + 1e-5)
        # depth[mask] = 0
        # depth = depth.repeat(3, -1)
        # plt.imshow(depth)
        # plt.show()

        mask = face_id >= 0
        positions = positions[mask].detach().cpu().numpy().reshape(-1, 3)

        try:
            indices = np.random.choice(len(positions), per_frame_n, replace=False)
        except:
            warnings.warn(f"Warning: {name} temporal error exception")
            indices = np.random.choice(len(positions), per_frame_n, replace=True)
        positions = positions[indices]

        all_positions.append(positions)

    all_positions = np.concatenate(all_positions, axis=0)

    # revert 
    all_positions = (all_positions / scale) + center

    # scene = trimesh.Scene([mesh, trimesh.PointCloud(all_positions)])
    # scene.show()

    return all_positions


def visualize_poses(poses, size=0.05, bound=1, mesh=None):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=[2*bound]*3).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    if mesh is not None:
        objects.append(mesh)

    scene = trimesh.Scene(objects)
    scene.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pred', type=str)
    parser.add_argument('gt', type=str)
    parser.add_argument('--data_path', type=int, default=2500000)
    parser.add_argument('--N', type=int, default=2500000)
    parser.add_argument('--scale', type=float, default=0.8)
    parser.add_argument('--fix_pred_coord', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    opt = parser.parse_args()
    
    setup_seed(opt.seed)
    pred_mesh = trimesh.load(opt.pred, force='mesh', skip_material=True, process=False)
    gt_mesh = trimesh.load(opt.gt, force='mesh', skip_material=True)

    name = os.path.basename(opt.gt).replace('.obj', '')
    

    # fix gt coord
    v = gt_mesh.vertices
    R = Rot.from_euler('x', 90, degrees=True)
    v = R.apply(v)
    gt_mesh.vertices = v

    # fix my scale
    if opt.scale != 1:
        v = pred_mesh.vertices # [N, 3]
        v /= opt.scale
        pred_mesh.vertices = v

    if opt.fix_pred_coord: # for nvdiffrec's output
        v = pred_mesh.vertices
        R = Rot.from_euler('x', 90, degrees=True)
        v = R.apply(v)
        pred_mesh.vertices = v

    # scene = trimesh.Scene([pred_mesh, gt_mesh])
    # scene.show()

    root_path = f'./data/nerf_synthetic/{name}/transforms_test.json'
    root_path = '/home/nsml/injae/data/Stanford-ORB_LDR/baking_scene001/transforms_train.json'

    with open(root_path, 'r') as f:
        transform = json.load(f)
    
    frames = np.array(transform["frames"])
    poses = []
    for f in frames:
        pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
        poses.append(pose)
    poses = np.stack(poses, axis=0)

    # visualize_poses(poses, mesh=gt_mesh)
    
    # load intrinsics
    if 'fl_x' in transform or 'fl_y' in transform:
        fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y'])
        fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x'])
    elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
        # blender, assert in radians. already downscaled since we use H/W
        fl_x = W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
        fl_y = H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
        if fl_x is None: fl_x = fl_y
        if fl_y is None: fl_y = fl_x
    else:
        raise RuntimeError('Failed to load focal length, please check the transforms.json!')

    cx = (transform['cx']) if 'cx' in transform else (W / 2.0)
    cy = (transform['cy']) if 'cy' in transform else (H / 2.0)
    
    intrinsics = np.array([fl_x, fl_y, cx, cy])

    gt_points = sample_surface(poses, intrinsics, gt_mesh, opt.N)
    pred_points = sample_surface(poses, intrinsics, pred_mesh, opt.N)

    if opt.vis:
        gt_color = np.array([[0, 0, 255]], dtype=np.uint8).repeat(len(gt_points), 0)
        pred_color = np.array([[255, 0, 0]], dtype=np.uint8).repeat(len(pred_points), 0)
        scene = trimesh.Scene([trimesh.PointCloud(pred_points, pred_color), trimesh.PointCloud(gt_points, gt_color)])
        scene.show()

    import time
    s = time.time()
    cd = chamfer_distance(pred_points, gt_points, direction='bi')
    print("Calculate time: ", time.time()-s)
    print(f'[CD] {name: <20} {cd:.6f}')
    
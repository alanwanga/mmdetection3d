from tqdm import tqdm
import pickle
import numpy as np
from pyquaternion import Quaternion

def _remove_close(points, radius=1.0):
    points_numpy = points
    x_filt = np.abs(points_numpy[:, 0]) < radius
    y_filt = np.abs(points_numpy[:, 1]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    return points[not_close]

data_root = "/mnt/ml"
sample_num = 28130
n_sweep = 9
data = pickle.load(open("/mnt/ml/nuscenes/train_sorted.pkl", "rb"))
for sample_idx in tqdm(range(sample_num)):
    # 9 sweeps, 4 pre, 5 post
    # find 9 sweeps for this sample
    #if sample_idx > 3:
    #    continue
    key_frame = data[sample_idx]
    key_frame_pcd = key_frame['lidar_path']
    sweeps = key_frame['sweeps']
    pts_filename = key_frame['lidar_path'].replace("./data", data_root)
    points = np.fromfile(pts_filename, dtype=np.float32)
    points = points.reshape(-1, 5)
    attribute_dims = None
    ts = key_frame['timestamp']
    sweep_points_list = [points]
    if len(sweeps) == 0:
        for i in range(n_sweep):
            sweep_points_list.append(_remove_close(points))
    else:
        if len(sweeps) <= n_sweep:
            choices = np.arange(len(sweeps))
        else:
            choices = np.random.choice(
                len(sweeps), n_sweep, replace=False)
        for idx in choices:
            sweep = sweeps[idx]
            points_sweep = np.fromfile(sweep['data_path'].replace('./data', data_root), dtype=np.float32)
            points_sweep = np.copy(points_sweep).reshape(-1, 5)
            points_sweep = _remove_close(points_sweep)
            sweep_ts = sweep['timestamp'] / 1e6
            points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                'sensor2lidar_rotation'].T
            points_sweep[:, :3] += sweep['sensor2lidar_translation']
            points_sweep[:, 4] = ts - sweep_ts
            sweep_points_list.append(points_sweep)
    #import ipdb
    #ipdb.set_trace()
    res = np.concatenate(sweep_points_list, axis=0)
    #res.tofile(open(pts_filename.replace("sample", "sample_9"), "w"))
    kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
    kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
    pcd = res.T
    pcd[:3,:] = np.dot(kitti_to_nu_lidar_inv.rotation_matrix, pcd[:3, :])
    pcd.T.tofile(open(pts_filename.replace('samples', "nusc_in_kitti"), "w"))

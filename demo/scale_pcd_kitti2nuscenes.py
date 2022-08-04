import numpy as np
from pyquaternion import Quaternion
import json
import pickle
from tqdm import tqdm

class Object3d(object):
    def __init__(self, label):
        self.h = float(label[5])
        self.w = float(label[3])
        self.l = float(label[4])
        self.ry = float(label[6])
        self.t = np.array((float(label[0]), float(
            label[1]), float(label[2])), dtype=np.float32)

def single_scale(x, src, dst, ratio=1):
    return x + (dst["mean"] - src["mean"]) * ratio

def get_scale_map(src, dst):
    return lambda x, ratio: (np.array([
        single_scale(x.l, src["length"], dst["length"], ratio),
        single_scale(x.h, src["height"], dst["height"], ratio),
        single_scale(x.w, src["width"], dst["width"], ratio),
    ]) / np.array([x.l, x.h, x.w])).reshape(1, 3)

def rescale_ptc(mapping, origin_ptc, labels, classes, rescale_classes=("car")):
    new_ptc = []
    mask = np.ones(origin_ptc.shape[0]).astype(bool)
    ratios = []
    ptc = origin_ptc[:, :3]
    for i, obj in enumerate(labels):
        if classes[i] in rescale_classes:
            R = np.array([[np.cos(obj.ry), 0, np.sin(obj.ry)],
                        [0, 1, 0],
                        [-np.sin(obj.ry), 0, np.cos(obj.ry)]])
            _ptc = np.dot(ptc - obj.t, R)
            _mask = (_ptc[:, 0] > -obj.l / 2.0) & (_ptc[:, 0] < obj.l / 2.0) & \
                (_ptc[:, 1] > -obj.h) & (_ptc[:, 1] < 0) & \
                (_ptc[:, 2] > -obj.w / 2.0) & (_ptc[:, 2] < obj.w / 2.0)
            ratio = 0
            if np.sum(_mask) > 0:
                mask[_mask] = False                
                ratio = 1
                tmp_ptc = _ptc[_mask] * mapping(obj, ratio)
                ptc_patch = np.dot(tmp_ptc, R.T) + obj.t
            #import ipdb;
            #ipdb.set_trace()
                patch_shape = ptc_patch.shape[0]
                padding = np.zeros((patch_shape, 2))
                new_ptc.append(np.hstack((ptc_patch, padding)))
            ratios.append(ratio)
    return np.concatenate(new_ptc + [origin_ptc[mask]], axis=0)

if __name__ == "__main__":

    data = pickle.load(open("/mnt/ml/nuscenes/train_sorted.pkl", "rb"))
    with open("/mnt/ml/nuscenes/label_stats_nusc.json") as f:
        src_label_stats = json.load(f)
    with open("/mnt/ml/nuscenes/label_stats_qualcomm.json") as f:
        dst_label_stats = json.load(f)
    mapping = get_scale_map(src_label_stats, dst_label_stats)
    sample_count = len(data)

    for i in tqdm(range(sample_count)):
        gt_boxes = data[i]['gt_boxes']
        kitti_pcd_path = data[i]['lidar_path'].replace('./data/nuscenes/samples', '/mnt/ml/nuscenes/nusc_in_kitti')
        ptc = np.fromfile(kitti_pcd_path, dtype=np.float32).reshape(-1,5)
        labels = [Object3d(object) for object in gt_boxes]
        # nx5
        points = rescale_ptc(mapping, ptc, labels, data[i]['gt_names'])
        pcd = points[:, :3].T
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        # 3xn
        pcd = np.dot(kitti_to_nu_lidar.rotation_matrix, pcd)
        points[:, :3] = pcd.T
        points = np.array(points, dtype=np.float32)
        points.tofile(open(data[i]['lidar_path'].replace('./data/nuscenes/samples', '/mnt/ml/nuscenes/sample_9s_scaled_nus'), 'w'))


    

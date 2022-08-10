import numpy as np
from pyquaternion import Quaternion
import json
import pickle
import copy
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

def rescale_ptc(origin_ptc, labels, classes, rescaled_classes=("car")):
    ratios = []
    ptc = origin_ptc[:, :3]
    for i, obj in enumerate(labels):
        if classes[i] in rescaled_classes:
            R = np.array([[np.cos(obj.ry), 0, np.sin(obj.ry)],
                            [0, 1, 0],
                            [-np.sin(obj.ry), 0, np.cos(obj.ry)]])
            _ptc = np.dot(ptc - obj.t, R)
            _mask = (_ptc[:, 0] > -obj.l / 2.0) & (_ptc[:, 0] < obj.l / 2.0) & \
                (_ptc[:, 1] > -obj.h) & (_ptc[:, 1] < 0) & \
                (_ptc[:, 2] > -obj.w / 2.0) & (_ptc[:, 2] < obj.w / 2.0)
            ratio = 0
            if np.sum(_mask) > 0:
                ratio = 1
            ratios.append(ratio)
    return ratios

def scale_labels(objs, mapping, ratios, classes, rescaled_classes=("car")):
    new_obj = []
    cnt = 0
    for i, obj in enumerate(objs):
        _obj = copy.deepcopy(obj)
        if classes[i] in rescaled_classes:
            l, h, w = (np.array([obj.l, obj.h, obj.w]) * mapping(obj, ratios[cnt]).reshape(-1)).tolist()
            _obj.l, _obj.h, _obj.w = l, h, w
            cnt += 1
        new_obj.append(_obj)
    return new_obj

if __name__ == "__main__":

    data = pickle.load(open("/mnt/ml/nuscenes/train_sorted.pkl", "rb"))
    with open("/mnt/ml/nuscenes/label_stats_nusc.json") as f:
        src_label_stats = json.load(f)
    with open("/mnt/ml/nuscenes/label_stats_qualcomm.json") as f:
        dst_label_stats = json.load(f)
    mapping = get_scale_map(src_label_stats, dst_label_stats)
    sample_count = len(data)

    for i in tqdm(range(sample_count)):
        #if i > 10:
        #    break
        kitti_pcd_path = data[i]['lidar_path'].replace('./data/nuscenes/samples', '/mnt/ml/nuscenes/nusc_in_kitti')
        ptc = np.fromfile(kitti_pcd_path, dtype=np.float32).reshape(-1,5)
        labels = [Object3d(object) for object in data[i]['gt_boxes']]
        ratios = rescale_ptc(ptc, labels, data[i]['gt_names'])
        labels = scale_labels(labels, mapping, ratios, data[i]['gt_names'])
        for j, label in enumerate(labels):
            data[i]['gt_boxes'][j][3] = label.w
            data[i]['gt_boxes'][j][4] = label.l
            data[i]['gt_boxes'][j][5] = label.h
    with open("/mnt/ml/nuscenes/train_sorted_rescaled.pkl", 'wb') as handle:
        pickle.dump(data, handle)


    

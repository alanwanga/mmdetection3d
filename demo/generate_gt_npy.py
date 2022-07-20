import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import math


def get_yaw(o):
    euler = R.from_euler(
        'yzx', [o['rotation']['y'], o['rotation']['z'], o['rotation']['x']], degrees=False)
    quat = euler.as_quat()
    q = dict()
    q["w"] = quat[3]
    q["x"] = quat[0]
    q["y"] = quat[1]
    q["z"] = quat[2]
    siny_cosp = 2 * (q["w"] * q["z"] + q["x"] * q["y"])
    cosy_cosp = 1 - 2 * (q["y"] * q["y"] + q["z"] * q["z"])
    yaw = math.atan2(siny_cosp, cosy_cosp)
    yaw = math.pi / 2 - yaw
    return yaw


def build_scene(annos):
    count = 0
    bbox_result = dict()
    bbox_result['items'] = []
    for idx, o in enumerate(annos['items']):
        count += 1
        bbox_result['items'].append(o)
    print(count)
    rbbox_array = np.zeros((len(bbox_result['items']), 9))
    rbbox_array[:, 0] = [-box['position']['y'] for box in bbox_result['items']]
    rbbox_array[:, 1] = [box['position']['x'] for box in bbox_result['items']]
    rbbox_array[:, 2] = [box['position']['z'] for box in bbox_result['items']]
    rbbox_array[:, 3] = [box['dimension']['x'] for box in bbox_result['items']]
    rbbox_array[:, 4] = [box['dimension']['y'] for box in bbox_result['items']]
    rbbox_array[:, 5] = [box['dimension']['z'] for box in bbox_result['items']]
    rbbox_array[:, 2] -= rbbox_array[:, 5] / 2
    rbbox_array[:, 6] = [get_yaw(box) for box in bbox_result['items']]
    return rbbox_array


data = json.load(open("/Users/yangxiaorui/Downloads/222.json", 'r'))
frame_id = 0
annos = data['frames'][frame_id]
gt_bboxes = build_scene(annos)
np.save(f"/Users/yangxiaorui/Downloads/frame_{frame_id}_gt", gt_bboxes)

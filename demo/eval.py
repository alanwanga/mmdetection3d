from scipy.spatial.transform import Rotation as R
import itertools

import argparse
import os
import json
from collections import defaultdict
from pathlib import Path

import math
import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import Polygon



'''
nuScenes: 
CLASSES = ('car': 0, 'truck': 1, 'trailer': 2, 'bus': 3, 'construction_vehicle': 4,
               'bicycle': 5, 'motorcycle': 6, 'pedestrian': 7, 'traffic_cone': 8,
               'barrier': 9)
qualcomm: 
    "Car": 1,
    "Bus": 2,
    "Debris": 3,
    "Motorcycle": 4,
    "Truck": 5,
    "Trailer": 6,
    "Traffic Sign": 7,
overlap classes: car: 0, truck: 1, trailer: 2, bus: 3, motorcycle: 6
'''

mapping = {
    0: 'Car',
    1: 'Truck',
    2: 'Trailer',
    3: 'Bus',
    6: 'Motorcycle'
}
class Box3D:
    """Data class used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(self, **kwargs):
        sample_token = kwargs["sample_token"]
        translation = kwargs["translation"]
        size = kwargs["size"]
        rotation = kwargs["rotation"]
        name = kwargs["name"]
        score = kwargs.get("score", -1)

        if not isinstance(sample_token, str):
            raise TypeError("Sample_token must be a string!")

        if not len(translation) == 3:
            raise ValueError("Translation must have 3 elements!")

        if np.any(np.isnan(translation)):
            raise ValueError("Translation may not be NaN!")

        if not len(size) == 3:
            raise ValueError("Size must have 3 elements!")

        if np.any(np.isnan(size)):
            raise ValueError("Size may not be NaN!")

        if not len(rotation) == 4:
            raise ValueError("Rotation must have 4 elements!")

        if np.any(np.isnan(rotation)):
            raise ValueError("Rotation may not be NaN!")

        if name is None:
            raise ValueError("Name cannot be empty!")

        # Assign.
        self.sample_token = sample_token
        self.translation = translation
        self.size = size
        self.volume = np.prod(self.size)
        self.score = score

        assert np.all([x > 0 for x in size])
        self.rotation = rotation
        self.name = name
        self.quaternion = Quaternion(self.rotation)

        self.width, self.length, self.height = size

        self.center_x, self.center_y, self.center_z = self.translation

        self.min_z = self.center_z - self.height / 2
        self.max_z = self.center_z + self.height / 2

        self.ground_bbox_coords = None
        self.ground_bbox_coords = self.get_ground_bbox_coords()

    @staticmethod
    def check_orthogonal(a, b, c):
        """Check that vector (b - a) is orthogonal to the vector (c - a)."""
        return np.isclose((b[0] - a[0]) * (c[0] - a[0]) + (b[1] - a[1]) * (c[1] - a[1]), 0)

    def get_ground_bbox_coords(self):
        if self.ground_bbox_coords is not None:
            return self.ground_bbox_coords
        return self.calculate_ground_bbox_coords()

    def calculate_ground_bbox_coords(self):
        """We assume that the 3D box has lower plane parallel to the ground.
        Returns: Polygon with 4 points describing the base.
        """
        if self.ground_bbox_coords is not None:
            return self.ground_bbox_coords

        rotation_matrix = self.quaternion.rotation_matrix

        cos_angle = rotation_matrix[0, 0]
        sin_angle = rotation_matrix[1, 0]

        point_0_x = self.center_x + self.length / \
            2 * cos_angle + self.width / 2 * sin_angle
        point_0_y = self.center_y + self.length / \
            2 * sin_angle - self.width / 2 * cos_angle

        point_1_x = self.center_x + self.length / \
            2 * cos_angle - self.width / 2 * sin_angle
        point_1_y = self.center_y + self.length / \
            2 * sin_angle + self.width / 2 * cos_angle

        point_2_x = self.center_x - self.length / \
            2 * cos_angle - self.width / 2 * sin_angle
        point_2_y = self.center_y - self.length / \
            2 * sin_angle + self.width / 2 * cos_angle

        point_3_x = self.center_x - self.length / \
            2 * cos_angle + self.width / 2 * sin_angle
        point_3_y = self.center_y - self.length / \
            2 * sin_angle - self.width / 2 * cos_angle

        point_0 = point_0_x, point_0_y
        point_1 = point_1_x, point_1_y
        point_2 = point_2_x, point_2_y
        point_3 = point_3_x, point_3_y

        assert self.check_orthogonal(point_0, point_1, point_3)
        assert self.check_orthogonal(point_1, point_0, point_2)
        assert self.check_orthogonal(point_2, point_1, point_3)
        assert self.check_orthogonal(point_3, point_0, point_2)

        self.ground_bbox_coords = Polygon(
            [
                (point_0_x, point_0_y),
                (point_1_x, point_1_y),
                (point_2_x, point_2_y),
                (point_3_x, point_3_y),
                (point_0_x, point_0_y),
            ]
        )

        return self.ground_bbox_coords

    def get_height_intersection(self, other):
        min_z = max(other.min_z, self.min_z)
        max_z = min(other.max_z, self.max_z)

        return max(0, max_z - min_z)

    def get_area_intersection(self, other) -> float:
        result = self.ground_bbox_coords.intersection(
            other.ground_bbox_coords).area

        assert result <= self.width * self.length

        return result

    def get_intersection(self, other) -> float:
        height_intersection = self.get_height_intersection(other)

        area_intersection = self.ground_bbox_coords.intersection(
            other.ground_bbox_coords).area

        return height_intersection * area_intersection

    def get_iou(self, other):
        intersection = self.get_intersection(other)
        union = self.volume + other.volume - intersection

        iou = np.clip(intersection / union, 0, 1)

        return iou

    def __repr__(self):
        return str(self.serialize())

    def serialize(self) -> dict:
        """Returns: Serialized instance as dict."""

        return {
            "sample_token": self.sample_token,
            "translation": self.translation,
            "size": self.size,
            "rotation": self.rotation,
            "name": self.name,
            "volume": self.volume,
            "score": self.score,
        }


def group_by_key(detections, key):
    groups = defaultdict(list)
    #import ipdb
    #ipdb.set_trace()
    for detection in detections:
        groups[detection[key]].append(detection)
    return groups


def wrap_in_box(input):
    result = {}
    for key, value in input.items():
        result[key] = [Box3D(**x) for x in value]

    return result


def get_envelope(precisions):
    """Compute the precision envelope.
    Args:
      precisions:
    Returns:
    """
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    return precisions


def get_ap(recalls, precisions):
    """Calculate average precision.
    Args:
      recalls:
      precisions: Returns (float): average precision.
    Returns:
    """
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    precisions = get_envelope(precisions)

    # to calculate area under PR curve, look for points where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap


def get_ious(gt_boxes, predicted_box):
    return [predicted_box.get_iou(x) for x in gt_boxes]


def recall_precision(all_gts, all_predictions, iou_threshold):
    all_gts = group_by_key(all_gts, 'category')
    all_predictions = group_by_key(all_predictions, "category")
    for cat in [0,1,3]:
        gt = all_gts[mapping[cat]]
        num_gts = len(gt)
        if num_gts == 0:
            #print(f"{mapping[cat]} no gt")
            continue
        image_gts = group_by_key(gt, "sample_token")
        image_gts = wrap_in_box(image_gts)

        sample_gt_checked = {sample_token: np.zeros(
            len(boxes)) for sample_token, boxes in image_gts.items()}
        
        predictions = all_predictions[cat]
        predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
        if len(predictions) == 0:
            #print(f"{mapping[cat]} no predictions")
            continue
        # go down dets and mark TPs and FPs
        num_predictions = len(predictions)
        tp = np.zeros(num_predictions)
        fp = np.zeros(num_predictions)

        for prediction_index, prediction in enumerate(predictions):
            predicted_box = Box3D(**prediction)

            sample_token = prediction["sample_token"]

            max_overlap = -np.inf
            jmax = -1

            try:
                gt_boxes = image_gts[sample_token]  # gt_boxes per sample
                gt_checked = sample_gt_checked[sample_token]  # gt flags per sample
            except KeyError:
                gt_boxes = []
                gt_checked = None

            if len(gt_boxes) > 0:
                overlaps = get_ious(gt_boxes, predicted_box)
                max_overlap = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if max_overlap > iou_threshold:
                if gt_checked[jmax] == 0:
                    tp[prediction_index] = 1.0
                    gt_checked[jmax] = 1
                    # print(prediction)
                    # print(max_overlap)
                    # print(jmax)
                else:
                    fp[prediction_index] = 1.0
            else:
                fp[prediction_index] = 1.0

        # compute precision recall
        fp = np.cumsum(fp, axis=0)
        tp = np.cumsum(tp, axis=0)
        #print(num_gts)
        #print(cat)
        recalls = tp / float(num_gts)

        assert np.all(0 <= recalls) & np.all(recalls <= 1)

        # avoid divide by zero in case the first detection matches a difficult ground truth
        precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        assert np.all(0 <= precisions) & np.all(precisions <= 1)

        ap = get_ap(recalls, precisions)
        #print(f"Cat: {mapping[cat]}, prediction {len(predictions)} gt {len(gt)}")
        print(f"Cat: {mapping[cat]}, IoU threshold: {iou_threshold:.2f}, Precsion: {precisions[-1]:.4f}, Recall: {recalls[-1]:.4f}, AP: {ap:.4f}")
    # return recalls, precisions, ap


def get_average_precisions(gt: list, predictions: list, class_names: list, iou_threshold: float) -> np.array:
    """Returns an array with an average precision per class.
    Args:
        gt: list of dictionaries in the format described below.
        predictions: list of dictionaries in the format described below.
        class_names: list of the class names.
        iou_threshold: IOU threshold used to calculate TP / FN
    Returns an array with an average precision per class.
    Ground truth and predictions should have schema:
    gt = [{
    'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
    'translation': [974.2811881299899, 1714.6815014457964, -23.689857123368846],
    'size': [1.796, 4.488, 1.664],
    'rotation': [0.14882026466054782, 0, 0, 0.9888642620837121],
    'name': 'car'
    }]
    predictions = [{
        'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
        'translation': [971.8343488872263, 1713.6816097857359, -25.82534357061308],
        'size': [2.519726579986132, 7.810161372666739, 3.483438286096803],
        'rotation': [0.10913582721095375, 0.04099572636992043, 0.01927712319721745, 1.029328402625659],
        'name': 'car',
        'score': 0.3077029437237213
    }]
    """
    assert 0 <= iou_threshold <= 1

    gt_by_class_name = group_by_key(gt, "name")
    pred_by_class_name = group_by_key(predictions, "name")

    average_precisions = np.zeros(len(class_names))

    for class_id, class_name in enumerate(class_names):
        if class_name in pred_by_class_name:
            recalls, precisions, average_precision = recall_precision(
                gt_by_class_name[class_name], pred_by_class_name[class_name], iou_threshold
            )
            average_precisions[class_id] = average_precision

    return average_precisions


def get_class_names(gt: dict) -> list:
    """Get sorted list of class names.
    Args:
        gt:
    Returns: Sorted list of class names.
    """
    return sorted(list(set([x["name"] for x in gt])))


def build_scene(file: str, score_thres=0.0, index=None, frameId=None, cats=None) -> list:
    scene = []
    with open(file) as f:
        data = json.load(f)
        for frame in data['frames']:
            items = []
            if 'items' not in frame:
                continue
            if index is not None and frame['frameId'] > index:
                continue
            for o in frame['items']:
                if 'score' in o and o['score'] < score_thres:
                    continue
                if cats is not None and o['category'] not in cats:
                    continue
                item = {}
                if frameId is not None:
                    item['sample_token'] = str(frameId)
                else:
                    item['sample_token'] = str(frame['frameId'])
                item['translation'] = [o['position']['x'],
                                       o['position']['y'], o['position']['z']]
                item['size'] = [o['dimension']['x'],
                                o['dimension']['y'], o['dimension']['z']]
                euler = R.from_euler(
                    'yzx', [o['rotation']['y'], o['rotation']['z'], o['rotation']['x']])
                quat = euler.as_quat()
                item['rotation'] = quat
                item['name'] = o['category']
                item['category'] = o['category']
                item['score'] = 1 if 'score' not in o else o['score']
                items.append(item)
            scene.append(items)
    return list(itertools.chain.from_iterable(scene))


def parse_args():
    parser = argparse.ArgumentParser(
        description='eval a model')
    # parser.add_argument('--gt_file', type=str, default='/Users/yangxiaorui/Downloads/anno_frame_6000_6150.json')
    # parser.add_argument('--pred_file', type=str, default='/Users/yangxiaorui/appen/code/mmdetection3d/demo/test.txt')
    # parser.add_argument('--pred_file', type=str, default='/Users/yangxiaorui/Downloads/frame_6000_pred.json')


    parser.add_argument('--gt_file', type=str, default='/home/ssm-user/xiaorui/lidar/qualcomm/20220704_Qualcomm_package/annotation/anno_frame_6000_6150.json')
    parser.add_argument('--pred_file', type=str, default='/home/ssm-user/xiaorui/lidar/qualcomm/pred/preds.txt')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    gt_file = args.gt_file
    gt = build_scene(gt_file, cats=['Car', 'Truck', 'Bus', 'Trailer', 'Motorcycle'])

    pred_file = args.pred_file
    if pred_file.endswith(".txt"):
        preds = []
        pred_files = [l.strip() for l in open(pred_file, "r").readlines()]
        for pred in pred_files:
            preds.extend(build_scene(pred, score_thres=0.3, frameId=int(os.path.basename(pred)[6:10]) - 6000, cats=[0,1,2,3,6]))
    else:
        preds = build_scene(pred_file, score_thres=0.3, frameId=int(os.path.basename(pred_file)[6:10]) - 6000, cats=[0,1,2,3,6])
    for iou_threshold in np.arange(0.1, 0.91, 0.1):
        iou_threshold = round(iou_threshold, 2)
        # for cat in ["Car", "Truck", "Trailer", "Bus", "Motorcycle"]
        # recalls, precisions, ap = recall_precision(gt, preds, iou_threshold)
        recall_precision(gt, preds, iou_threshold)

        

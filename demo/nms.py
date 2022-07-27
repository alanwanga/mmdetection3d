#!/usr/bin/env python
# coding=utf-8

from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
import torch
import json
import numpy as np
from mmdet3d.core import (xywhr2xyxyr)
from tqdm import tqdm
#frame_id = 80
# json_path = f"/Users/yangxiaorui/Downloads/interval_10/frame_{6000 + frame_id}/frame_{6000 + frame_id}_pred.json"
# pred_bboxes = np.load(f"/Users/yangxiaorui/Downloads/frame_{6000 + frame_id}_pred.npy")
#json_path = "/home/ssm-user/xiaorui/lidar/qualcomm/pred/test/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20200930_201619-67c8496f/frame_6080/frame_6080_pred.json"
#pred_bboxes = np.load("/home/ssm-user/xiaorui/lidar/qualcomm/pred/test/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20200930_201619-67c8496f/frame_6080/frame_6080_pred.npy")

#res = open("/home/ssm-user/xiaorui/lidar/qualcomm/pred/cp/nms_result.txt", "w")
#preds = [l.strip() for l in open("/home/ssm-user/xiaorui/lidar/qualcomm/pred/cp/cp_pred.txt", "r").readlines()]
preds = ["/home/ssm-user/xiaorui/lidar/qualcomm/pred/cp/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20200930_201619-67c8496f/frame_6080/frame_6080_pred.npy"]
for pred_npy in tqdm(preds):
    json_path = pred_npy.replace("npy", "json")
    pred_bboxes = np.load(pred_npy)


    jj = json.load(open(json_path, "r"))
    score_thres = 0.3
    valid_idxs = []
    top_scores = []
    for i in range(len(jj['frames'][0]['items'])):
        if jj['frames'][0]['items'][i]['score'] < score_thres:
            continue
        valid_idxs.append(i)
        top_scores.append(jj['frames'][0]['items'][i]['score'])

    box_preds = torch.zeros((len(valid_idxs), 5))
    box_preds[:, 0] = torch.from_numpy(pred_bboxes[valid_idxs, 1])
    box_preds[:, 1] = -torch.from_numpy(pred_bboxes[valid_idxs, 0])
    box_preds[:, 2] = torch.from_numpy(pred_bboxes[valid_idxs, 4])
    box_preds[:, 3] = torch.from_numpy(pred_bboxes[valid_idxs, 3])
    box_preds[:, 4] = torch.from_numpy(pred_bboxes[valid_idxs, 6])

    #print(box_preds[0])
    print(len(valid_idxs))
    print(torch.from_numpy(np.array(top_scores, dtype=np.float32)).cuda().shape)
    boxes_for_nms = xywhr2xyxyr(box_preds)
# the nms in 3d detection just remove overlap boxes.
    selected = nms_gpu(
        boxes_for_nms.cuda(),
        torch.from_numpy(np.array(top_scores, dtype=np.float32)).cuda(),
        thresh=0.7,
        pre_maxsize=1000,
        post_max_size=1000)
#    res.write(f"{','.join([str(i) for i in selected.tolist()])}\n")
    print(top_scores)
    for idx in selected:
        print(top_scores[idx])
    print(selected.sort())


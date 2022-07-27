# Code is referenced from https://github.com/tianweiy/CenterPoint/blob/cb25e870b271fe8259e91c5d17dcd429d74abc91/det3d/core/bbox/box_np_ops.py
# Any question may contact Xiaorui Yang: xyang@appen.com
import numpy as np
from pypcd import pypcd
import json
import numba
from scipy.spatial.transform import Rotation as R
import argparse
from tqdm import tqdm

def corners_nd(dims):
    """generate relative box corners based on length per dim and
    origin point.
    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    # 3
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners = dims.reshape([-1, 1, ndim]) * \
        corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners


def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.
    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]):
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array(
        [
            [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
            [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
            [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
            [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
            [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
            [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
        ]
    ).transpose([2, 0, 1, 3])
    return surfaces


def center_to_corner_box3d(centers, dims, angles=None):
    """convert kitti locations, dimensions and angles to corners
    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    corners = corners_nd(dims)
    # corners: [N, 8, 3]
    corners -= dims.reshape([-1, 1, 3]) / 2
    if angles is not None:
        obj_count = corners.shape[0]
        for idx in range(obj_count):
            r = R.from_euler('YZX', angles[idx], degrees=False)
            inter = r.apply(corners[idx, :])
            corners[idx, :] = inter
    corners += centers.reshape([-1, 1, 3])
    return corners


@numba.njit
def surface_equ_3d_jitv2(surfaces):
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    num_polygon = surfaces.shape[0]
    max_num_surfaces = surfaces.shape[1]
    normal_vec = np.zeros(
        (num_polygon, max_num_surfaces, 3), dtype=surfaces.dtype)
    d = np.zeros((num_polygon, max_num_surfaces), dtype=surfaces.dtype)
    sv0 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    sv1 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    for i in range(num_polygon):
        for j in range(max_num_surfaces):
            sv0[0] = surfaces[i, j, 0, 0] - surfaces[i, j, 1, 0]
            sv0[1] = surfaces[i, j, 0, 1] - surfaces[i, j, 1, 1]
            sv0[2] = surfaces[i, j, 0, 2] - surfaces[i, j, 1, 2]
            sv1[0] = surfaces[i, j, 1, 0] - surfaces[i, j, 2, 0]
            sv1[1] = surfaces[i, j, 1, 1] - surfaces[i, j, 2, 1]
            sv1[2] = surfaces[i, j, 1, 2] - surfaces[i, j, 2, 2]
            normal_vec[i, j, 0] = sv0[1] * sv1[2] - sv0[2] * sv1[1]
            normal_vec[i, j, 1] = sv0[2] * sv1[0] - sv0[0] * sv1[2]
            normal_vec[i, j, 2] = sv0[0] * sv1[1] - sv0[1] * sv1[0]

            d[i, j] = (
                -surfaces[i, j, 0, 0] * normal_vec[i, j, 0]
                - surfaces[i, j, 0, 1] * normal_vec[i, j, 1]
                - surfaces[i, j, 0, 2] * normal_vec[i, j, 2]
            )
    return normal_vec, d


@numba.njit
def _points_in_convex_polygon_3d_jit(
    points, polygon_surfaces, normal_vec, d, num_surfaces=None
):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def points_in_convex_polygon_3d_jit(points, polygon_surfaces, num_surfaces=None):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jitv2(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_in_convex_polygon_3d_jit(
        points, polygon_surfaces, normal_vec, d, num_surfaces
    )


def points_in_rbbox(points, rbbox):
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6:])
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points, surfaces)
    return indices


# def cuboid2seg(pcd_path, bbox_result):
#     pc = pypcd.PointCloud.from_path(pcd_path)
#
#     items = []
#     for _item in bbox_result['items']:
#         items.append({
#             "id": _item["id"],
#             "category": _item["category"],
#             "number": _item["number"]+1000,
#             "points": []
#         })
#     return items


def cuboid2seg(pcd_path, bbox_result):
    pc = pypcd.PointCloud.from_path(pcd_path)

    point_count = len(pc.pc_data['x'])
    points_array = np.zeros((point_count, 3))
    points_array[:, 0] = pc.pc_data['x']
    points_array[:, 1] = pc.pc_data['y']
    points_array[:, 2] = pc.pc_data['z']

    rbbox_count = len(bbox_result['items'])
    rbbox_array = np.zeros((rbbox_count, 9))
    rbbox_array[:, 0] = [box['position']['x'] for box in bbox_result['items']]
    rbbox_array[:, 1] = [box['position']['y'] for box in bbox_result['items']]
    rbbox_array[:, 2] = [box['position']['z'] for box in bbox_result['items']]
    rbbox_array[:, 3] = [box['dimension']['x'] for box in bbox_result['items']]
    rbbox_array[:, 4] = [box['dimension']['y'] for box in bbox_result['items']]
    rbbox_array[:, 5] = [box['dimension']['z'] for box in bbox_result['items']]
    rbbox_array[:, 6] = [box['rotation']['y'] for box in bbox_result['items']]
    rbbox_array[:, 7] = [box['rotation']['z'] for box in bbox_result['items']]
    rbbox_array[:, 8] = [box['rotation']['x'] for box in bbox_result['items']]

    res = points_in_rbbox(points_array, rbbox_array)
    for idx in range(rbbox_count):
        bbox_result['items'][idx]['point_count'] = np.count_nonzero(res[:, idx])
    return bbox_result

def parse_args():
    parser = argparse.ArgumentParser(
        description='eval a model')
    # parser.add_argument('--gt_file', type=str, default='/Users/yangxiaorui/Downloads/anno_frame_6000_6150.json')
    # parser.add_argument('--pred_file', type=str, default='/Users/yangxiaorui/appen/code/mmdetection3d/demo/test.txt')
    # parser.add_argument('--pred_file', type=str, default='/Users/yangxiaorui/Downloads/frame_6000_pred.json')


    parser.add_argument('--gt_file', type=str, default='/home/ssm-user/xiaorui/lidar/qualcomm/20220704_Qualcomm_package/annotation/anno_frame_6000_6150.json')
    # parser.add_argument('--pred_file', type=str, default='/home/ssm-user/xiaorui/lidar/qualcomm/pred/preds.txt')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    #gt_file = args.gt_file
    gt_file = "/home/ssm-user/xiaorui/lidar/qualcomm/20220704_Qualcomm_package/annotation/anno_frame_6000_6150.json"
    pcd_root = "/Users/yangxiaorui/Downloads/"
    pcd_root = "/home/ssm-user/xiaorui/lidar/qualcomm/20220704_Qualcomm_package/point_pcd/"
    #gt_all = json.load(open("/Users/yangxiaorui/Downloads/anno_frame_6000_6150.json", "r"))['frames']
    gt_all = json.load(open(gt_file, "r"))['frames']
    new_gt_all = []
    for frame_id in tqdm(range(150)):
        pcd_file = f'{pcd_root}frame_{6000 + frame_id}.pcd'
        gt = gt_all[frame_id]
        new_gt = cuboid2seg(pcd_file, gt)
        new_gt_all.append(new_gt)
    output_path = gt_file.replace(".json", "_new.json")
    with open(output_path, "w") as f:
        json.dump(new_gt_all, f)

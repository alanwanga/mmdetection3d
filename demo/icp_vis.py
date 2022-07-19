import copy
import numpy as np

try:
    import open3d as o3d
    from open3d import geometry
except ImportError:
    raise ImportError(
        'Please run "pip install open3d" to install open3d first.')


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size, source, target):

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    # draw_registration_result(source_down, target_down, np.identity(4))

    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f"
          % distance_threshold)
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def _draw_points(points,
                 vis,
                 points_size=2,
                 point_color=(0.5, 1, 0),
                 mode='xyz',
                 second_points=None):
    """Draw points on visualizer.

    Args:
        points (numpy.array | torch.tensor, shape=[N, 3+C]):
            points to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_size (int): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float]): the color of points.
            Default: (0.5, 0.5, 0.5).
        mode (str):  indicate type of the input points, avaliable mode
            ['xyz', 'xyzrgb']. Default: 'xyz'.

    Returns:
        tuple: points, color of each point.
    """
    vis.get_render_option().point_size = points_size  # set points size

    points = points.copy()
    pcd = geometry.PointCloud()
    if mode == 'xyz':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = np.tile(np.array(point_color), (points.shape[0], 1))
    elif mode == 'xyzrgb':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = points[:, 3:6]
        # normalize to [0, 1] for open3d drawing
        if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
            points_colors /= 255.0
    else:
        raise NotImplementedError

    pcd.colors = o3d.utility.Vector3dVector(points_colors)

    if second_points is not None:
        points = second_points.copy()
        pcd2 = geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points[:, :3])
        point_color = (0, 0, 1)
        points_colors = np.tile(np.array(point_color), (points.shape[0], 1))
        pcd2.colors = o3d.utility.Vector3dVector(points_colors)
        voxel_size = 1
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
            voxel_size, pcd, pcd2)
        result_fast = execute_fast_global_registration(source_down, target_down,
                                                       source_fpfh, target_fpfh,
                                                       voxel_size)

        print("fast")
        print(result_fast.transformation)
        draw_registration_result(source_down, target_down,
                                 result_fast.transformation)
        distance_threshold = voxel_size * 0.4

        result_icp = o3d.registration.registration_icp(
            source_down, target_down, distance_threshold, result_fast.transformation,
            o3d.registration.TransformationEstimationPointToPlane())
        draw_registration_result(
            source_down, target_down, result_icp.transformation)
        print("icp")
        print(result_icp.transformation)

    return pcd, points_colors


class Visualizer(object):
    r"""Online visualizer implemented with Open3d.

    Args:
        points (numpy.array, shape=[N, 3+C]): Points to visualize. The Points
            cloud is in mode of Coord3DMode.DEPTH (please refer to
            core.structures.coord_3d_mode).
        bbox3d (numpy.array, shape=[M, 7]): 3d bbox (x, y, z, dx, dy, dz, yaw)
            to visualize. The 3d bbox is in mode of Box3DMode.DEPTH with
            gravity_center (please refer to core.structures.box_3d_mode).
            Default: None.
        save_path (str): path to save visualized results. Default: None.
        points_size (int): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float]): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float]): the color of bbox. Default: (0, 1, 0).
        points_in_box_color (tuple[float]):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int): rotation axis of bbox. Default: 2.
        center_mode (bool): indicate the center of bbox is bottom center
            or gravity center. avaliable mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str):  indicate type of the input points, avaliable mode
            ['xyz', 'xyzrgb']. Default: 'xyz'.
    """

    def __init__(self,
                 points,
                 second_points=None,
                 points_size=2,
                 point_color=(0.5, 1, 0),
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz'):
        super(Visualizer, self).__init__()
        assert 0 <= rot_axis <= 2
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

        # init visualizer
        self.o3d_visualizer = o3d.visualization.Visualizer()
        self.o3d_visualizer.create_window()
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0])  # create coordinate frame
        self.o3d_visualizer.add_geometry(mesh_frame)

        self.points_size = points_size
        self.point_color = point_color
        self.bbox_color = bbox_color
        self.points_in_box_color = points_in_box_color
        self.rot_axis = rot_axis
        self.center_mode = center_mode
        self.mode = mode

        # draw points
        if points is not None:
            self.pcd, self.points_colors = _draw_points(
                points, self.o3d_visualizer, points_size, point_color, mode, second_points)

    def show(self, save_path=None):
        """Visualize the points cloud.

        Args:
            save_path (str): path to save image. Default: None.
        """

        self.o3d_visualizer.run()

        if save_path is not None:
            self.o3d_visualizer.capture_screen_image(save_path)

        self.o3d_visualizer.destroy_window()
        return


# points: n x 5
# dtype = np.dtype('B')
# Qualcomm
numpy_data = np.load("/Users/yangxiaorui/Downloads/frame_6000_points.npy")
second_data = np.load("/Users/yangxiaorui/Downloads/frame_6003_points.npy")
vis = Visualizer(numpy_data[::10, :], second_data[::10, :])
vis.show(None)

# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.


import os
import pickle
import numpy as np
from ...utils import common_utils
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.label_pb2 import Label
try:
    tf.enable_eager_execution()
except:
    pass

WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']

def transform_bbox_waymo(label: Label) -> np.ndarray:
    """Transform object's 3D bounding box using Waymo utils"""
    heading = -label.box.heading
    bbox_corners = get_bbox(label)
 
    mat = transform_utils.get_yaw_rotation(heading)
    rot_mat = mat.numpy()[:2, :2]
 
    return bbox_corners @ rot_mat
 
def get_bbox(label: Label) -> np.ndarray:
    width, length = label.box.width, label.box.length
    return np.array([[-0.5 * length, -0.5 * width],
                     [-0.5 * length, 0.5 * width],
                     [0.5 * length, -0.5 * width],
                     [0.5 * length, 0.5 * width]])

def transform_point_homogeneous(p: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform a 3D point using a 4x4 transformation matrix."""
    p_homogeneous = np.hstack((p, [1]))
    p_transformed = transform @ p_homogeneous
    return p_transformed[:3]

def build_open3d_bbox_extrinsic(box: np.ndarray, label: Label, extrinsic_matrix: np.ndarray) -> list:
    """Create bounding box's points and lines needed for drawing in open3d, including extrinsic transformation."""
    x, y, z = label.box.center_x, label.box.center_y, label.box.center_z
    z_bottom = z - label.box.height / 2
    z_top = z + label.box.height / 2

    transformed_points = []
    for idx in range(box.shape[0]):
        # Apply original transformation to get bottom and top points
        bottom_point = np.array([x + box[idx, 0], y + box[idx, 1], z_bottom])
        top_point = np.array([x + box[idx, 0], y + box[idx, 1], z_top])
        
        # Transform points with extrinsic matrix
        bottom_point_transformed = transform_point_homogeneous(bottom_point, extrinsic_matrix)
        top_point_transformed = transform_point_homogeneous(top_point, extrinsic_matrix)
        
        transformed_points.extend([bottom_point_transformed, top_point_transformed])

    return transformed_points

def is_point_inside_bbox(point, bbox_min, bbox_max):
    """Check if a point is inside the bounding box."""
    return np.all(point >= bbox_min) and np.all(point <= bbox_max)

def filter_bboxes_with_point_cloud(bboxes, point_cloud):
    """Filters bounding boxes to keep those that have at least one point from the point cloud within them."""
    filtered_bboxes_indices = []
    for i, bbox in enumerate(bboxes):
        bbox_points = np.array(bbox)
        bbox_min = bbox_points.min(axis=0)
        bbox_max = bbox_points.max(axis=0)
        point_count = 0
        for point in point_cloud:
            if is_point_inside_bbox(point, bbox_min, bbox_max):
                point_count += 1
        if point_count >= 15:  # Assuming a threshold of 15 points to consider the bbox valid
            filtered_bboxes_indices.append((i, point_count))
    return filtered_bboxes_indices
    
def generate_labels(frame, point_cloud, extrinsic_matrix, pose):
    obj_name, difficulty, dimensions, locations, heading_angles = [], [], [], [], []
    tracking_difficulty, speeds, accelerations, obj_ids = [], [], [], []
    num_points_in_gt = []
    laser_labels = frame.laser_labels

    bboxes = []
    for label in laser_labels:
        bbox_corners = transform_bbox_waymo(label)
        bbox_points = build_open3d_bbox_extrinsic(bbox_corners, label, extrinsic_matrix)      
        bboxes.append(bbox_points)

    filtered_bboxes_info = filter_bboxes_with_point_cloud(bboxes, point_cloud)
    filtered_bboxes_indices  = [info[0] for info in filtered_bboxes_info]
    num_of_lidar_pts = [info[1] for info in filtered_bboxes_info]


    j = 0
    for i in filtered_bboxes_indices:
        box = laser_labels[i].box
        class_ind = laser_labels[i].type
        loc = [box.center_x, box.center_y, box.center_z]
        heading_angles.append(box.heading)
        obj_name.append(WAYMO_CLASSES[class_ind])
        difficulty.append(laser_labels[i].detection_difficulty_level)
        tracking_difficulty.append(laser_labels[i].tracking_difficulty_level)
        dimensions.append([box.length, box.width, box.height])  # lwh in unified coordinate of OpenPCDet
        locations.append(loc)
        obj_ids.append(laser_labels[i].id)
        num_points_in_gt.append(num_of_lidar_pts[j])
        j += 1 
        speeds.append([laser_labels[i].metadata.speed_x, laser_labels[i].metadata.speed_y])
        accelerations.append([laser_labels[i].metadata.accel_x, laser_labels[i].metadata.accel_y])

    annotations = {}
    annotations['name'] = np.array(obj_name)
    annotations['difficulty'] = np.array(difficulty)
    annotations['dimensions'] = np.array(dimensions)
    annotations['location'] = np.array(locations)
    annotations['heading_angles'] = np.array(heading_angles)

    annotations['obj_ids'] = np.array(obj_ids)
    annotations['tracking_difficulty'] = np.array(tracking_difficulty)
    annotations['num_points_in_gt'] = np.array(num_points_in_gt)
    annotations['speed_global'] = np.array(speeds)
    annotations['accel_global'] = np.array(accelerations)

    annotations = common_utils.drop_info_with_name(annotations, name='unknown')
    if annotations['name'].__len__() > 0:
        global_speed = np.pad(annotations['speed_global'], ((0, 0), (0, 1)), mode='constant', constant_values=0)  # (N, 3)
        speed = np.dot(global_speed, np.linalg.inv(pose[:3, :3].T))
        speed = speed[:, :2]
        
        gt_boxes_lidar = np.concatenate([
            annotations['location'], annotations['dimensions'], annotations['heading_angles'][..., np.newaxis], speed],
            axis=1
        )
    else:
        gt_boxes_lidar = np.zeros((0, 9))
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    return annotations


def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1)):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)

    for c in calibrations:
        points_single, cp_points_single, points_NLZ_single, points_intensity_single, points_elongation_single \
            = [], [], [], [], []
        for cur_ri_index in ri_index:
            range_image = range_images[c.name][cur_ri_index]
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_NLZ = range_image_tensor[..., 3]
            range_image_intensity = range_image_tensor[..., 1]
            range_image_elongation = range_image_tensor[..., 2]
            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.where(range_image_mask))
            points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
            points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
            points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
            cp = camera_projections[c.name][0]
            cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))

            points_single.append(points_tensor.numpy())
            cp_points_single.append(cp_points_tensor.numpy())
            points_NLZ_single.append(points_NLZ_tensor.numpy())
            points_intensity_single.append(points_intensity_tensor.numpy())
            points_elongation_single.append(points_elongation_tensor.numpy())

        points.append(np.concatenate(points_single, axis=0))
        cp_points.append(np.concatenate(cp_points_single, axis=0))
        points_NLZ.append(np.concatenate(points_NLZ_single, axis=0))
        points_intensity.append(np.concatenate(points_intensity_single, axis=0))
        points_elongation.append(np.concatenate(points_elongation_single, axis=0))

    return points, cp_points, points_NLZ, points_intensity, points_elongation


def save_lidar_points(frame, cur_save_path, extrinsic_matrix, use_two_returns=True, fov_min_angle=120, fov_max_angle=170):
    ret_outputs = frame_utils.parse_range_image_and_camera_projection(frame)
    if len(ret_outputs) == 4:
        range_images, camera_projections, seg_labels, range_image_top_pose = ret_outputs
    else:
        assert len(ret_outputs) == 3
        range_images, camera_projections, range_image_top_pose = ret_outputs

    points, cp_points, points_in_NLZ_flag, points_intensity, points_elongation = convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1) if use_two_returns else (0,)
    )

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_in_NLZ_flag = np.concatenate(points_in_NLZ_flag, axis=0).reshape(-1, 1)
    points_intensity = np.concatenate(points_intensity, axis=0).reshape(-1, 1)
    points_elongation = np.concatenate(points_elongation, axis=0).reshape(-1, 1)

    # Apply the extrinsic transformation if provided
    point_cloud_homogeneous = np.hstack((points_all, np.ones((points_all.shape[0], 1))))
    transformed_points = point_cloud_homogeneous @ extrinsic_matrix.T
    
    # Calculate azimuth angles (in degrees)
    azimuth_angles = np.degrees(np.arctan2(transformed_points[:, 1], transformed_points[:, 0]))
    
    # Filter for the specified FoV
    fov_filter = (azimuth_angles >= fov_min_angle) & (azimuth_angles <= fov_max_angle)
    points_all = transformed_points[fov_filter, :-1]  # Exclude the homogeneous coordinate
    points_in_NLZ_flag = points_in_NLZ_flag[fov_filter]
    points_intensity = points_intensity[fov_filter]
    points_elongation = points_elongation[fov_filter]

    num_points_of_each_lidar = points_all.shape[0]
    # Concatenate filtered points with their intensity, elongation, and NLZ flag
    save_points = np.concatenate([
        points_all, points_intensity, points_elongation, points_in_NLZ_flag
    ], axis=-1).astype(np.float32)

    np.save(cur_save_path, save_points)
    # print('saving to ', cur_save_path)
    return num_points_of_each_lidar, points_all  # Since the points are now filtered, return the count of filtered points

def process_single_sequence(sequence_file, save_path, sampled_interval, has_label=True, use_two_returns=True, update_info_only=False):
    sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]

    # print('Load record (sampled_interval=%d): %s' % (sampled_interval, sequence_name))
    if not sequence_file.exists():
        print('NotFoundError: %s' % sequence_file)
        return []

    dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
    cur_save_dir = save_path / sequence_name
    cur_save_dir.mkdir(parents=True, exist_ok=True)
    pkl_file = cur_save_dir / ('%s.pkl' % sequence_name)

    sequence_infos = []
    if pkl_file.exists():
        sequence_infos = pickle.load(open(pkl_file, 'rb'))
        sequence_infos_old = None
        if not update_info_only:
            print('Skip sequence since it has been processed before: %s' % pkl_file)
            return sequence_infos
        else:
            sequence_infos_old = sequence_infos
            sequence_infos = []

    for cnt, data in enumerate(dataset):
        if cnt % sampled_interval != 0:
            continue
        # print(sequence_name, cnt)
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        extrinsic_matrix = np.array(frame.context.laser_calibrations[4].extrinsic.transform, dtype=np.float32).reshape(4,4)
        
        info = {}
        pc_info = {'num_features': 5, 'lidar_sequence': sequence_name, 'sample_idx': cnt}
        info['point_cloud'] = pc_info

        info['frame_id'] = sequence_name + ('_%03d' % cnt)
        info['metadata'] = {
            'context_name': frame.context.name,
            'timestamp_micros': frame.timestamp_micros
        }
        image_info = {}
        for j in range(5):
            width = frame.context.camera_calibrations[j].width
            height = frame.context.camera_calibrations[j].height
            image_info.update({'image_shape_%d' % j: (height, width)})
        info['image'] = image_info

        pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        info['pose'] = pose

        if update_info_only and sequence_infos_old is not None:
            assert info['frame_id'] == sequence_infos_old[cnt]['frame_id']
            num_points_of_each_lidar = sequence_infos_old[cnt]['num_points_of_each_lidar']
        else:
            num_points_of_each_lidar, point_cloud = save_lidar_points(
                frame, cur_save_dir / ('%04d.npy' % cnt), extrinsic_matrix, use_two_returns=use_two_returns
            )


        if has_label:
            annotations = generate_labels(frame, point_cloud, extrinsic_matrix, pose=pose)
            info['annos'] = annotations
    
        info['num_points_of_each_lidar'] = num_points_of_each_lidar

        sequence_infos.append(info)

    with open(pkl_file, 'wb') as f:
        pickle.dump(sequence_infos, f)

    print('Infos are saved to (sampled_interval=%d): %s' % (sampled_interval, pkl_file))
    return sequence_infos



import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

import numpy as np
import cv2
from numpy.linalg import inv


chess_size = 1  # mm
extrinsic_param = os.path.join(PARENT_DIR, 'params/E1.npz')

with np.load(extrinsic_param) as X:
    mtx, dist, Mat, tvecs = [X[i] for i in ('mtx', 'dist', 'Mat', 'tvec')]

tvec = tvecs * chess_size
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]


def size_interpolation(img, image_shape):
    """resize the image to the desired shape

    Parameters
    ----------
    img : ndarray
        original image needs to be shaped
    image_shape : tuple
        goal image shape
    Returns
    -------
    ndarray
        resized image
    """
    instack = img.transpose((1, 2, 0))
    mask = cv2.resize(instack, (image_shape.shape[1], image_shape.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    mask = mask.transpose((2, 0, 1))
    return mask


def depth_filter(depth_data, mask):
    """filtered the depth data and extract the instance depth value from 
    the depth iamge

    Parameters
    ----------
    depth_data : ndarray
        depth value
    mask : ndarray
        instance segmentation mask result

    Returns
    -------
    ndarray
        instance depth value
    """

    depth_data[np.isnan(depth_data)] = -1
    depth_mask = depth_data * mask
    depth_mask[(depth_mask < 0.5) | (depth_mask > 2)] = 0
    return depth_mask


def yolo_pose(yolo_pose_results):
    """Extract the yolo pose estimation information

    Parameters
    ----------
    yolo_pose_results : ultralytics return
        yolo pose result

    Returns
    -------
    points: ndarray
        the second and third keypoints in homogeneous format
    visual: ndarray
        the second and third keypoints visualbility
    """
    confidence_thresdhold = 0.7
    confidence = yolo_pose_results[0].keypoints.conf.detach().cpu().numpy()
    indices = np.where(confidence > confidence_thresdhold)
    keypoints = yolo_pose_results[0].keypoints.data.detach().cpu().numpy()
    num_object = len(indices[0])
    points = keypoints[:, 1:3, :2]
    visual = keypoints[:, 1:3, 2]

    one_vector = np.ones(points.shape[:2])
    one_vector = np.expand_dims(one_vector, axis=2)
    points = np.concatenate((points, one_vector), axis=2)
    return points, visual


def pose_seg_match(points, depth_mask):
    """match instance seg depth and pos instance

    Parameters
    ----------
    points : ndarray
        keypoints 
    depth_mask : ndarray
        instance seg depth

    Returns
    -------
    depth: ndarray
        object keypoints depth [num_object] 
    """
    object_num = points.shape[0]
    num_points = 10
    thres_num = 3
    depth = np.zeros(object_num)
    for i in range(object_num):
        point1 = points[i, 0, :2]
        point2 = points[i, 1, :2]
        x1, y1 = point1
        x2, y2 = point2
        x_points = np.linspace(x1, x2, num_points).astype(int)
        y_points = np.linspace(y1, y2, num_points).astype(int)
        x_points = np.clip(x_points, 0, depth_mask.shape[2] - 1)
        y_points = np.clip(y_points, 0, depth_mask.shape[1] - 1)
        pts_on_line = np.vstack((x_points, y_points))
        depth_mask_line = depth_mask[:, pts_on_line[1], pts_on_line[0]]
        valid_depth_count = np.count_nonzero(depth_mask_line, axis=1)
        index = np.argmax(valid_depth_count)
        if valid_depth_count[index] >= thres_num:
            depth_match = depth_mask_line[index, :]
            depth_match = camera_to_world(pts_on_line, depth_match)
            depth[i] = np.sum(depth_match) / valid_depth_count[index]
    return depth


def camera_to_world(pts_on_line, depth_match):
    """convert the point from camera frame to world frame

    Parameters
    ----------
    pts_on_line : ndarray
        keypoints
    depth_match : ndarray
        depth of the keypoints

    Returns
    -------
    world_depth: ndarray
        depth in the world coordinate frame
    """
    img_x = pts_on_line[0]
    img_y = pts_on_line[1]
    x = (img_x - cx) * (depth_match) / fx
    y = (img_y - cy) * (depth_match) / fy
    camera_coord = np.vstack((x, y, depth_match))
    world_coord = inv(Mat) @ (camera_coord - tvec)
    world_depth = world_coord[-1, :]
    world_depth[(world_depth > 0.5) | (world_depth < -0.5)] = 0
    return world_depth


def projection(points, mtx, Mat, tvecs, depth=0):
    """camera back projection

    Parameters
    ----------
    points : ndarray
        keypoints for camera back projection
    mtx : ndarray
        camera intrinsic parameter
    Mat : ndarray
        camera extrinsic parameter: rotation matrix
    tvecs : ndarray
        camera extrinsic parameter: translation vector
    depth : int, optional
        depth of the point in the world coordinate frame, by default 0

    Returns
    -------
    result: ndarray
        point in the world coordinate frame
    """
    num_object = points.shape[0]
    results = np.zeros_like(points)
    for i in range(num_object):
        point = points[i, :].reshape(3, 1)
        point2 = inv(mtx) @ point
        predefined_z = depth[i // 2]
        # predefined_z = 0
        vec_z = Mat[:, [2]] * predefined_z
        Mat2 = np.copy(Mat)
        Mat2[:, [2]] = -1 * point2
        vec_o = -1 * (vec_z + tvecs)
        result = inv(Mat2) @ vec_o
        results[i] = result.squeeze()
    return results


def result_project(point, depth):
    """project the keypoint from camera frame to world coordinate frame

    Parameters
    ----------
    point : ndarray
        THe second and third keypoints
    depth : ndarray
        Depth in the world cooridnate frame

    Returns
    -------
    grap_point: ndarray
        grasp point in the world coordinate frame 
    orientation: ndarray
        green onion orientation in the world coordinate frame
    """
    point = point.reshape((-1, 3))
    result = projection(point, mtx, Mat, tvec, depth)
    result = np.round(result, 2)
    result = result.reshape((-1, 2, 3))
    grasp_point = np.mean(result, axis=1)
    p1 = result[:, 0, :]
    p2 = result[:, 1, :]
    dy = p2[:, 1] - p1[:, 1]
    dx = p2[:, 0] - p1[:, 0]
    orientation = np.arctan2(dy, dx) * 180 / np.pi
    grasp_point[:, -1] = depth
    return grasp_point, orientation

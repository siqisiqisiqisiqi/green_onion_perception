import numpy as np
import cv2


chess_size = 1  # mm

with np.load('params/E1.npz') as X:
    mtx, dist, Mat, tvecs, rvecs = [X[i]
                                    for i in ('mtx', 'dist', 'Mat', 'tvecs', 'rvecs')]
tvec = tvecs * chess_size
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]


def drawing(img, points, results, orientation):
    """visualize the perception and localization result

    Parameters
    ----------
    img : ndarray
        real time image
    points : ndarray
        keypoints in the image reference frame
    results : ndarray
        keypoints in the world coordinate frame
    orientation : ndarray
        object orientation in the world coordinate frame

    Returns
    -------
    char
        User input to quit the visualization
    """
    points = np.mean(points, axis=1)
    axis = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.1]]).reshape(-1, 3)
    origin = np.float32([[0, 0, 0]]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs / chess_size, mtx, dist)
    corner, _ = cv2.projectPoints(origin, rvecs, tvecs / chess_size, mtx, dist)
    corner = tuple(corner[0].astype(int).ravel())
    img = cv2.line(img, corner, tuple(
        imgpts[0].astype(int).ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(
        imgpts[1].astype(int).ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(
        imgpts[2].astype(int).ravel()), (0, 0, 255), 5)

    cv2.putText(img, 'X axis', (corner[0] + 50, corner[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, 'Y axis', (corner[0] + 10, corner[1] + 80), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2, cv2.LINE_AA)

    object_num = points.shape[0]
    for i in range(object_num):
        point = points[i].reshape(3, 1).astype(int)
        result = results[i].reshape(3, 1)
        orient = np.round(orientation[i], 2)
        cv2.putText(img, f'[{result[0,0]:.3f}, {result[1,0]:.3f}, {orient:.2f}]', (point[0, 0] + 50, point[1,
                    0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 0, 128), 2, cv2.LINE_AA)
        cv2.circle(img, (point[0, 0], point[1, 0]), 5, (128, 0, 128), 5)

        # cv2.imwrite(f'image/image5_improved.jpg', img)
    cv2.imshow('img', img)
    k = cv2.waitKey(10)
    return k

#!/home/grail/.pyenv/versions/yolo_ultralytics/bin/python
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)

import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO

from utils.util import size_interpolation, depth_filter, yolo_pose
from utils.util import pose_seg_match, result_project
from green_onion_perception.msg import Keypoints, Pose
from src.test_visual import drawing


class Camera:
    def __init__(self):
        rospy.init_node("camera", anonymous=True)

        self.bridge = CvBridge()
        rospy.Subscriber("zed2i/zed_node/rgb/image_rect_color",
                         Image, self.get_image)

        rospy.Subscriber("zed2i/zed_node/depth/depth_registered",
                         Image, self.get_depth_image)

        self.kpts_pub = rospy.Publisher('kpts_pose', Keypoints, queue_size=10)

        self.camera_info = rospy.wait_for_message(
            "zed2i/zed_node/rgb/camera_info", CameraInfo)

        self.model_seg = YOLO("./weights/green_onion_instance_seg.pt")
        self.model_pose = YOLO("./weights/green_onion_skeleton.pt")

        self.cv_image = None
        self.depth_image = None

        rospy.wait_for_message("zed2i/zed_node/rgb/image_rect_color", Image)
        rospy.wait_for_message("zed2i/zed_node/depth/depth_registered", Image)

        self.visual = rospy.get_param("~visual")

        self.image_shape = self.cv_image.shape
        self.rate = rospy.Rate(10)

    def get_image(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.image_t = 1e-9 * data.header.stamp.nsecs + data.header.stamp.secs
        except CvBridgeError as e:
            print(e)

    def get_depth_image(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
            # TODO: Debug the inaccurate depth problem
            self.depth_image = self.depth_image - 0.045
        except CvBridgeError as e:
            print(e)

    def pub_kpts_pose(self, pro_result, orientation):
        num_obj = len(orientation)
        kpts_data = Keypoints()
        kpts_data.num = num_obj
        kpts_data.stamp = rospy.Time.now()
        for i in range(num_obj):
            kpt_pose = Pose()
            kpt_pose.x, kpt_pose.y, kpt_pose.z = pro_result[i]
            kpt_pose.yaw = orientation[i]
            kpts_data.grap_pose.append(kpt_pose)
        self.kpts_pub.pub(kpts_data)

    def run(self):
        rospy.sleep(2)

        while not rospy.is_shutdown():
            if self.cv_image is None:
                continue
            # visualize the image
            cv2.imshow("image", self.depth_image)
            cv2.waitKey(2)

            # model inference
            seg_results = self.model_seg(
                self.cv_image[:, :, :3], verbose=False)
            pose_results = self.model_pose(
                self.cv_image[:, :, :3], verbose=False)

            # instance seg depth calculation
            mask_result = seg_results[0].masks.data.cpu().detach().numpy()
            mask = size_interpolation(mask_result, self.image_shape)
            depth_mask = depth_filter(self.depth_image, mask)

            # keypoints calculation
            points, _ = yolo_pose(pose_results)

            # match instance seg depth and keypoints
            depth = pose_seg_match(points, depth_mask)

            # project the points to the world coordinate frame
            pro_result, orientation = result_project(points, depth)

            # publish the message to the topic
            self.pub_kpts_pose(pro_result, orientation)

            # publish the keypoint pose
            self.pub_kpts_pose(pro_result, orientation)

            # visualization
            if self.visual:
                k = drawing(self.cv_image, points, pro_result, orientation)
                if k == ord('q'):
                    rospy.loginfo("Complete!")
                    break

            rospy.loginfo("This is a test")
            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = Camera()
        node.run()
    except rospy.ROSInterruptException:
        pass
    rospy.loginfo("Finish the code!")

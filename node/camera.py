#!/home/grail/.pyenv/versions/yolo_ultralytics/bin/python
import rospy
import pandas as pd
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from numpy.linalg import inv
from ultralytics import YOLO


class Camera:
    def __init__(self):
        rospy.init_node("camera", anonymous=True)

        self.bridge = CvBridge()
        rospy.Subscriber("zed2i/zed_node/rgb/image_rect_color",
                         Image, self.get_image)

        rospy.Subscriber("zed2i/zed_node/depth/depth_registered",
                         Image, self.get_depth_image)

        self.camera_info = rospy.wait_for_message(
            "zed2i/zed_node/rgb/camera_info", CameraInfo)

        self.cv_image = None
        self.depth_image = None

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

        except CvBridgeError as e:
            print(e)

    def run(self):
        rospy.sleep(2)

        while not rospy.is_shutdown():
            if self.cv_image is None:
                continue
            # visualize the image
            cv2.imshow("image", self.depth_image)
            cv2.waitKey(2)

            rospy.loginfo("This is a test")
            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = Camera()
        node.run()
    except rospy.ROSInterruptException:
        pass
    rospy.loginfo("Finish the code!")

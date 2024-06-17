#!/usr/bin/env python3

"""
ROS1 node for Aruco marker pose estimation.
Subscribes to:
   /zed2i/zed_node/image_rect_raw (sensor_msgs/Image)
   /zed2i/zed_node/depth_rect_raw (sensor_msgs/Image)

Publishes to:
    /aruco_poses (geometry_msgs/PoseArray)
    /aruco_markers (aruco_msgs/ArucoMarkers)
    /aruco_image (sensor_msgs/Image)

Parameters:
    marker_size - size of the markers in meters (default 0.065)
    aruco_dictionary_id - dictionary that was used to generate markers (default DICT_5X5_250)
    camera_frame - camera optical frame to use (default "camera_depth_optical_frame")
"""

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
import message_filters

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray
from aruco_pose_estimation.msg import ArucoMarkers
from dt_apriltags import Detector

from utils import ARUCO_DICT
from pose_estimation import pose_estimation

class ArucoNode:
    def __init__(self):
        rospy.init_node('aruco_node', anonymous=True)

        self.bridge = CvBridge()
        self.use_depth_input = rospy.get_param('~use_depth_input', False)
        self.marker_size = rospy.get_param('~marker_size', 0.150)
        self.dictionary_id_name = rospy.get_param('~aruco_dictionary_id', 'DICT_APRILTAG_25h9')
        self.camera_frame = rospy.get_param('~camera_frame', '/zed2i/zed_node/left/camera_info')
        
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/zed2i/zed_node/left/camera_info')
        self.image_topic = rospy.get_param('~image_topic', '/zed2i/zed_node/left/image_rect_color')
        self.depth_image_topic = rospy.get_param('~depth_image_topic', '/zed2i/zed_node/depth/depth_registered')

        self.poses_pub = rospy.Publisher('/aruco_poses', PoseArray, queue_size=10)
        self.markers_pub = rospy.Publisher('/aruco_markers', ArucoMarkers, queue_size=10)
        self.image_pub = rospy.Publisher('/aruco_image', Image, queue_size=10)
        
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None
        self.camera_info = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.info_callback)

        try:
            dictionary_id = cv2.aruco.__getattribute__(self.dictionary_id_name)
            if dictionary_id not in ARUCO_DICT.values():
                raise AttributeError
        except AttributeError:
            rospy.logerr("bad aruco_dictionary_id: {}".format(self.dictionary_id_name))
            options = "\n".join([s for s in ARUCO_DICT])
            rospy.logerr("valid options: {}".format(options))

        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)
        
        # Duckietown AR tag detector 
        self.at_detector = Detector(searchpath=['apriltags'],
                       families='tag25h9',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.5,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
        
        
        if (bool(self.use_depth_input)):
            self.image_sub = message_filters.Subscriber(self.image_topic, Image)
            self.depth_image_sub = message_filters.Subscriber(self.depth_image_topic, Image)
            self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_image_sub], 10, 0.05)
            self.ts.registerCallback(self.rgb_depth_sync_callback)
        else:
            self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
            
    def info_callback(self, info_msg):
        self.info_msg = info_msg
        # Get the intrinsic matrix and distortion coefficients from the camera info
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.K), (3, 3))
        self.distortion = np.array(self.info_msg.D)

        rospy.loginfo("Camera info received.")
        rospy.loginfo("Intrinsic matrix: {}".format(self.intrinsic_mat))
        rospy.loginfo("Distortion coefficients: {}".format(self.distortion))
        rospy.loginfo("Camera frame: {}x{}".format(self.info_msg.width, self.info_msg.height))

        # Assume that camera parameters will remain the same...
        self.camera_info.unregister()
        
    def image_callback(self, rgb_msg):
        cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')

        markers = ArucoMarkers()
        pose_array = PoseArray()

        markers.header.frame_id = self.info_msg.header.frame_id
        pose_array.header.frame_id = self.info_msg.header.frame_id
        markers.header.stamp = rgb_msg.header.stamp
        pose_array.header.stamp = rgb_msg.header.stamp

        # self.intrinsic_mat = np.reshape([522.9013061523438, 0.0, 646.6828002929688, 0.0, 522.9013061523438, 367.96405029296875, 0.0, 0.0, 1.0], (3, 3))
        # self.distortion = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        frame, pose_array, markers = pose_estimation(
            rgb_frame=cv_image,
            depth_frame= None,
            aruco_detector=self.aruco_detector,
            marker_size=self.marker_size,
            matrix_coefficients=self.intrinsic_mat,
            distortion_coefficients=self.distortion,
            pose_array=pose_array,
            markers=markers
        )

        if len(markers.marker_ids) > 0:
            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))


    def rgb_depth_sync_callback(self, rgb_msg, depth_msg):
        cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
        cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')

        markers = ArucoMarkers()
        pose_array = PoseArray()

        markers.header.frame_id = self.info_msg.header.frame_id
        pose_array.header.frame_id = self.info_msg.header.frame_id
        markers.header.stamp = rgb_msg.header.stamp
        pose_array.header.stamp = rgb_msg.header.stamp

        self.intrinsic_mat = np.reshape([615.95431, 0., 325.26983, 0., 617.92586, 257.57722, 0., 0., 1.], (3, 3))
        self.distortion = np.array([0.142588, -0.311967, 0.003950, -0.006346, 0.000000])

        frame, pose_array, markers = pose_estimation(
            rgb_frame=cv_image,
            depth_frame=cv_depth_image,
            aruco_detector=self.aruco_detector,
            marker_size=self.marker_size,
            matrix_coefficients=self.intrinsic_mat,
            distortion_coefficients=self.distortion,
            pose_array=pose_array,
            markers=markers
        )

        if len(markers.marker_ids) > 0:
            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

if __name__ == '__main__':
    try:
        node = ArucoNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python 
import rospy
import math
import tf
from tf.msg import tfMessage 
from tf.transformations import quaternion_from_euler 
import geometry_msgs.msg
from geometry_msgs.msg import Point, Quaternion, TransformStamped

def tf_callback(msg):
    for transform in msg.transforms:
    	if transform.header.frame_id == "map" and transform.child_frame_id == "odom":
    		print("transfrom from map to odom")
    		print(transform)
    	if transform.header.frame_id == "odom" and transform.child_frame_id == "base_link":
    		print("transfrom from odom to baselink")
    		odom_to_zedbase = transform
    		print(transform)
    		
    	zedbase_to_rearaxle = (50, 100, 0, 0, 0, 0) # x, y, z, roll, pitch, yaw 
    	
    	translation = zedbase_to_rearaxle[:3]
    	rotation_quat = quaternion_from_euler(*zedbase_to_rearaxle[3:])
    	
    	rear_axle_transform = TransformStamped()
    	rear_axle_transform.header.stamp = rospy.Time.now()
    	rear_axle_transform.header.frame_id = "base_link"
    	rear_axle_transform.child_frame_id = "car_rear_axle"
    	rear_axle_transform.transform.translation = translation 
    	rear_axle_transform.transform.rotation = Quaternion(*rotation_quat)
    	
    	tf_broadcaster.sendTransform(translation, rotation_quat, rospy.Time.now(), "car_rear_axle", "base_link")
    	
    		
if __name__ == '__main__':
    rospy.init_node('tf_listener')
    listener = tf.TransformListener()
    tf_sub = rospy.Subscriber('/tf', tfMessage, tf_callback)
    tf_broadcaster = tf.TransformBroadcaster()
    rospy.spin()



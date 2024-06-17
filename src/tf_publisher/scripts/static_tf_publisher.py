#!/usr/bin/env python

import rospy
import tf
import geometry_msgs.msg

def publish_static_transform():
    rospy.init_node('static_tf_publisher')

    br = tf.TransformBroadcaster()

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        now = rospy.Time.now()

        translation = (-1.780, -0.310, -1.150)  # x, y, z
        rotation = (0.0, 0.0, 0.0, 1)  # quaternion x, y, z, w

        br.sendTransform(
            translation,
            rotation,
            now,
            "rear_axle_link",  # child frame
            "zed2i_base_link"  # parent frame
            )

        rospy.loginfo("Static transform published from zed2i_base_link to rear_axle_link")

        rate.sleep()

if __name__ == '__main__':
    try:
        publish_static_transform()
    except rospy.ROSInterruptException:
        pass

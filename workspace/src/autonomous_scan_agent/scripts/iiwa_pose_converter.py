#!/usr/bin/env python3
import rospy
from iiwa_msgs.msg import CartesianPose
from geometry_msgs.msg import PoseStamped

class IiwaPoseConverter:
    def __init__(self):
        rospy.init_node('iiwa_pose_converter', anonymous=True)

        # Publish to /iiwa/pose_stamped
        self.pub = rospy.Publisher('/iiwa/pose_stamped', PoseStamped, queue_size=1)

        # Subscribe to /iiwa/state/CartesianPose
        self.sub = rospy.Subscriber('/iiwa/state/CartesianPose', CartesianPose, self.callback)

        rospy.loginfo("iiwa_pose_converter started: converting /iiwa/state/CartesianPose -> /iiwa/pose_stamped")

    def callback(self, msg):
        # msg is iiwa_msgs/CartesianPose
        # it has a field 'poseStamped' which is geometry_msgs/PoseStamped
        # We just forward it
        if msg.poseStamped:
            self.pub.publish(msg.poseStamped)

if __name__ == '__main__':
    try:
        IiwaPoseConverter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

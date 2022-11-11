#date: 2022-11-11T17:13:34Z
#url: https://api.github.com/gists/53bd8ede440c58ed35e574706e91b403
#owner: https://api.github.com/users/saltysushi8

#!/usr/bin/env python3

import rospy
import math 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

class Lab2:


    def __init__(self):
        """
        Class constructor
        """
        ### REQUIRED CREDIT
        ### Initialize node, name it 'lab2'
        rospy.init_node('lab2')
        
        ### Tell ROS that this node publishes Twist messages on the '/cmd_vel' topic
        
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        ### Tell ROS that this node subscribes to Odometry messages on the '/odom' topic
        ### When a message is received, call self.update_odometry
        rospy.Subscriber('/odom', Odometry , self.update_odometry)
        
        ### Tell ROS that this node subscribes to PoseStamped messages on the '/move_base_simple/goal' topic
        ### When a message is received, call self.go_to
        rospy.Subscriber('move_base_simple/goal', PoseStamped , self.go_to)
        self.px = 0
        self.py = 0
        self.pth = 0
    

    def send_speed(self, linear_speed, angular_speed):
        """
        Sends the speeds to the motors.
        :param linear_speed  [float] [m/s]   The forward linear speed.
        :param angular_speed [float] [rad/s] The angular speed for rotating around the body center.
        """
        ### REQUIRED CREDIT
        ### Make a new Twist message
        msg_cmd_vel = Twist()
        # Linear velocity
        msg_cmd_vel.linear.x = linear_speed
        msg_cmd_vel.linear.y = 0.0
        msg_cmd_vel.linear.z = 0.0
        #Angular velocity
        msg_cmd_vel.angular.x = 0.0
        msg_cmd_vel.angular.y = 0.0
        msg_cmd_vel.angular.z = angular_speed
        
        ### Publish the message
        print("publishing message")
        self.cmd_vel.publish(msg_cmd_vel)
        
        

    def drive(self, distance, linear_speed):
        """
        Drives the robot in a straight line.
        :param distance     [float] [m]   The distance to cover.
        :param linear_speed [float] [m/s] The forward linear speed.
        """
        ### REQUIRED CREDIT
        # rospy.sleep(.05)
        # initPose = self.px
        # toleranceDis = .01 #[m]

        # while (abs(initPose - self.px)) < (distance - toleranceDis):
        #     self.send_speed(linear_speed, 0)
        #     rospy.sleep(.05)
        # self.send_speed(0, 0)
        
        initPosex = self.px
        initPosey = self.py
        #initDis = math.sqrt(initPosex ** 2 + initPosey ** 2)
        toleranceDis = .01
        
        while (abs(math.sqrt((self.px - initPosex) ** 2 + (self.py - initPosey) ** 2)) < distance):
            print('self.py:', self.py)
            self.send_speed(linear_speed, 0)
            rospy.sleep(.05)
        self.send_speed(0, 0)



    def rotate(self, angle, aspeed):
        """
        Rotates the robot around the body center by the given angle.
        :param angle         [float] [rad]   The distance to cover.
        :param angular_speed [float] [rad/s] The angular speed.
        """
        ### REQUIRED CREDIT
        rospy.sleep(.05)
        initRot = self.pth
        toleranceAngle = .1
        targetAngle = initRot + angle
        print('initial: ', initRot)
        
        if (targetAngle > math.pi):
            targetAngle = targetAngle - 2*math.pi
        
        if (targetAngle < -math.pi):
            targetAngle = targetAngle + 2*math.pi
            
        # if (angle > 0):     #When the angle is positive
        #     while (angle > 3.14):   # and the angle is greater than 180 degrees
        #         angle = angle - 2*3.14  # The angle will be transfered to the same angle, but notated in negative
        # else:               # when the angle is negative
        #     while (angle < -3.14): # and the angle is less that -180 degrees
        #         angle = angle + 2*3.14 # The angle will be transfered to the same angle, but notated in positive
                
        if (targetAngle - angle < 0): 
            aspeed = -aspeed

        while (abs(targetAngle - self.pth)) > toleranceAngle:
               
            print('self.pth: ', self.pth)
            print('initpose: ', initRot)
            self.send_speed(0, aspeed)
            rospy.sleep(.05)

        self.send_speed(0, 0)



    def go_to(self, msg):
        """
        Calls rotate(), drive(), and rotate() to attain a given pose.
        This method is a callback bound to a Subscriber.
        :param msg [PoseStamped] The target pose.
        """
        targetTheta1 = math.atan2((msg.pose.position.y - self.py),(msg.pose.position.x- self.px))
        print('theta1:', targetTheta1)
        distance = math.sqrt((msg.pose.position.y - self.py) ** 2 + (msg.pose.position.x - self.px) **2)
        print('distance:', distance)

        quat_orig = msg.pose.orientation
        quat_list = [quat_orig.x, quat_orig.y, quat_orig.z, quat_orig.w]
        (roll , pitch , yaw) = euler_from_quaternion(quat_list)
        targetTheta2 = yaw
        print('theta2:', targetTheta2)

        self.rotate(targetTheta1 - self.pth, 1)
        rospy.sleep(0.05)
        self.drive(distance, .1)
        rospy.sleep(0.05)
        self.rotate(targetTheta2 - self.pth, 1)
        
        ### REQUIRED CREDIT
       



    def update_odometry(self, msg):
        """
        Updates the current pose of the robot.
        This method is a callback bound to a Subscriber.
        :param msg [Odometry] The current odometry information.
        """
        ### REQUIRED CREDIT
        self.px = msg.pose.pose.position.x
        self.py = msg.pose.pose.position.y
        quat_orig = msg.pose.pose.orientation
        quat_list = [quat_orig.x, quat_orig.y, quat_orig.z, quat_orig.w]
        (roll , pitch , yaw) = euler_from_quaternion(quat_list)
        self.pth = yaw
        
        


    def arc_to(self, position):
        """
        Drives to a given position in an arc.
        :param msg [PoseStamped] The target pose.
        """
        ### EXTRA CREDIT
        # TODO
        pass # delete this when you implement your code



    def smooth_drive(self, distance, linear_speed):
        """
        Drives the robot in a straight line by changing the actual speed smoothly.
        :param distance     [float] [m]   The distance to cover.
        :param linear_speed [float] [m/s] The maximum forward linear speed.
        """
        ### EXTRA CREDIT
        # TODO
        pass # delete this when you implement your code



    def run(self):
  
        #self.drive(.5, 0.1)
        #self.rotate(3.14/2, 0.1)
        
        #self.send_speed(0.1, 0)
        rospy.spin()
        

if __name__ == '__main__':
    Lab2().run()
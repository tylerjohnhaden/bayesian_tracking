#!/usr/bin/env python

"""Detect balls in camera view"""

import imutils
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import cv2
import numpy as np

ball_hsv_thresholds = {
    #'Purple': {'lower': (113, 35, 40), 'upper': (145, 160, 240)},
     'Blue': {'lower': (95, 150, 80), 'upper': (100, 255, 250)},
    # 'Green': {'lower': (43, 60, 40), 'upper': (71, 240, 200)},
    # 'Yellow': {'lower': (19, 60, 100), 'upper': (23, 255, 255)},
    # 'Orange': {'lower': (11, 150, 100), 'upper': (16, 255, 250)},
}

marker_rgb_colors = {c: (float(r) / 255, float(g) / 255, float(b) / 255)
                     for c, (r, g, b) in {
                         'Purple': (238, 130, 238),
                         'Blue': (0, 0, 255),
                         'Green': (0, 255, 0),
                         'Yellow': (255, 255, 0),
                         'Orange': (255, 140, 0),
                     }.items()
                     }


class BallDetector:
    def __init__(self):
        rospy.init_node('ball_detector')

        self.bridge = CvBridge()

        self.ball_image_pub = rospy.Publisher('/ball_image', Image, queue_size=5)
        self.ball_cov_pubs = {c: rospy.Publisher('/ballxyz/' + c, PoseWithCovarianceStamped, queue_size=5) for c in
                              ball_hsv_thresholds.keys()}
        self.ball_marker_pubs = {c: rospy.Publisher('/ball_marker/' + c, Marker, queue_size=5) for c in
                                 ball_hsv_thresholds.keys()}
        self.ball_arrow_pubs = {c: rospy.Publisher('/ball_velocity/' + c, Marker, queue_size=5) for c in
                                 ball_hsv_thresholds.keys()}
        self.xyz_yaw_pitch_distance = {c: [None for _ in range(30)] for c in ball_hsv_thresholds.keys()}
        self.bearing = {c: 0 for c in ball_hsv_thresholds.keys()}
        self.speed = {c: 0 for c in ball_hsv_thresholds.keys()}

        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)

        self.kernel = np.ones((5, 5), np.uint8)

        self.last_depth = None

        self.height, self.width = None, None  # pixels
        self.horizontal_half_angle = 0.436332  # radians
        self.vertical_half_angle = 0.375246  # radians

    def run(self):
        dt = 1
        hz = 1.0 / dt
        rate = rospy.Rate(hz)

        while not rospy.is_shutdown():
            for color in ball_hsv_thresholds.keys():
                self.calculate_bearing_and_speed(color)
                self.publish_pose_w_cov(color)
                self.publish_velocity(color)
            rate.sleep()

    def get_covariance(self, yaw, pitch, distance):
        """yaw and pitch range from -1 to 1, distance is strictly positive"""

        x_var = 0.015 + (0.12 * np.exp(distance - 4))
        y_var = 0.004 + ((yaw * 0.85) ** 20)
        z_var = 0.004 + ((pitch * 0.85) ** 20)

        return [
            x_var, 0, 0, 0, 0, 0,
            0, y_var, 0, 0, 0, 0,
            0, 0, z_var, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ]

    def transform(self, pixel_x, pixel_y, distance):
        yaw_scaled = (((pixel_x * 2) / self.width) - 1)
        pitch_scaled = -1 * (((pixel_y * 2) / self.height) - 1)

        horizontal_theta = yaw_scaled * self.horizontal_half_angle
        vertical_theta = pitch_scaled * self.vertical_half_angle

        y = - distance * np.cos(vertical_theta) * np.sin(horizontal_theta)
        z = distance * np.sin(vertical_theta) * np.cos(horizontal_theta)
        x = distance * np.cos(vertical_theta) * np.cos(horizontal_theta)

        return x, y, z, yaw_scaled, pitch_scaled

    def image_callback(self, image_msg):
        if self.last_depth is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(e)
            print(e)
            return

        if self.width is None:
            self.height, self.width, _ = cv_image.shape

        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        for color, thresholds in ball_hsv_thresholds.items():
            found_ball_xyz = None
            # use color to find ball candidates
            mask = cv2.inRange(hsv_image, thresholds['lower'], thresholds['upper'])
            smooth_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

            # use contours to filter noise
            contours = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            better_contours = imutils.grab_contours(contours)
            sorted_contours = sorted(better_contours, key=cv2.contourArea, reverse=True)

            # only select largest one
            sorted_contours = sorted_contours[:1]
            for i, contour in enumerate(sorted_contours):
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                ball_d = float(self.last_depth[int(y), int(x)]) / 1000

                # ignore if too close
                if ball_d > 0.1:
                    error = self.squared_mean_error(ball_d, radius)
                    if radius > 10 and error < 200:
                        tx, ty, tz, tyaw, tpitch = self.transform(x, y, ball_d)
                        found_ball_xyz = (tx, ty, tz, tyaw, tpitch, ball_d)

                        self.publish_marker(tx, ty, tz, color)
                        cv2.circle(hsv_image, (int(x), int(y)), int(radius), (0, 0, 0), 4)

        self.update_coordinates(found_ball_xyz, color)

        bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        try:
            img_msg_out = self.bridge.cv2_to_imgmsg(bgr_image)
            self.ball_image_pub.publish(img_msg_out)

        except CvBridgeError as e:
            rospy.logerr(e)
    
    def update_coordinates(self, xyz_yaw_pitch_distance, color):
        self.xyz_yaw_pitch_distance[color].pop(0)
        self.xyz_yaw_pitch_distance[color].append(xyz_yaw_pitch_distance)
    
    def calculate_bearing_and_speed(self, color):
        locations = []
        
        for i, history in enumerate(self.xyz_yaw_pitch_distance[color]):
            if history is not None:
                x, y, z, _, _, _ = history
                locations.append((i, (x, y, z)))
        
        if len(locations) < 2:
            self.speed[color] = 0
            return
        
        speeds = []
        bearings = []
        
        for i in range(len(locations) - 1):
            m, (x, y, z) = locations[i]
            n, (a, b, c) = locations[i + 1]
            
            speed = 30 * np.sqrt((a - x) ** 2 + (b - y) ** 2) / abs(m - n)
            bearing = np.arctan2(b - y, a - x)
            
            speeds.append(speed)
            bearings.append(bearing)
        
        self.speed[color] = np.average(speeds)
        self.bearing[color] = np.average(bearings)
        # print('{:.2f} (m/s), {:.2f} (radians)'.format(self.speed[color], self.bearing[color]))
        
        

    def depth_callback(self, image_msg):
        try:
            self.last_depth = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            rospy.logerr(e)
            print(e)

    def squared_mean_error(self, distance, radius):
        """filter out objects too large or too small"""
        return (radius - (6.37 + (38.33 / distance))) ** 2

    def publish_pose_w_cov(self, color):
        pwcs = PoseWithCovarianceStamped()
        pwcs.header.stamp = rospy.get_rostime()
        pwcs.header.frame_id = 'camera_link'
        pwcs.pose.pose.orientation.x = 0
        pwcs.pose.pose.orientation.y = 0
        pwcs.pose.pose.orientation.z = 0
        pwcs.pose.pose.orientation.w = 1
        
        xyz_y_p_d = next(
            (a for a in reversed(self.xyz_yaw_pitch_distance[color]) if a is not None), 
            None
        )

        if xyz_y_p_d is not None:
            x, y, z, yaw, pitch, distance = xyz_y_p_d
            pwcs.pose.pose.position.x = x
            pwcs.pose.pose.position.y = y
            pwcs.pose.pose.position.z = z
            pwcs.pose.covariance = self.get_covariance(yaw, pitch, distance)
        else:
            pwcs.pose.pose.position.x = 0
            pwcs.pose.pose.position.y = 0
            pwcs.pose.pose.position.z = 0
            inf_cov = 10000
            pwcs.pose.covariance = [
                inf_cov, 0, 0, 0, 0, 0,
                0, inf_cov, 0, 0, 0, 0,
                0, 0, inf_cov, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ]

        self.ball_cov_pubs[color].publish(pwcs)

    def publish_velocity(self, color):
        marker = Marker()
        marker.header.frame_id = '/camera_link'
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.ns = color + 'Velocity'
        marker.id = 0
        marker.scale.x = self.speed[color]
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0
        marker.color.g = 1
        marker.color.b = 0
        
        ox, oy, oz, ow = quaternion_from_euler(0, 0, self.bearing[color])
        
        marker.pose.orientation.x = ox
        marker.pose.orientation.y = oy
        marker.pose.orientation.z = oz
        marker.pose.orientation.w = ow
        
        xyz_y_p_d = next(
            (a for a in reversed(self.xyz_yaw_pitch_distance[color]) if a is not None), 
            None
        )

        if xyz_y_p_d is not None:
            x, y, z, _, _, _ = xyz_y_p_d
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
        else:
            marker.pose.position.x = 0
            marker.pose.position.y = 0
            marker.pose.position.z = 0
        
        
        self.ball_arrow_pubs[color].publish(marker)

    def publish_marker(self, x, y, z, color, diameter=0.15):
        r, g, b = marker_rgb_colors[color]

        marker = Marker()
        marker.header.frame_id = '/camera_link'
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.ns = color + 'Ball'
        marker.id = 0
        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = diameter
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z

        self.ball_marker_pubs[color].publish(marker)


if __name__ == '__main__':
    BallDetector().run()

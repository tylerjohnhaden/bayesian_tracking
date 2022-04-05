#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler


class EKF():
    def __init__(self, nk, dt, X, U, color='Blue'):
        self.nk = nk
        self.dt = dt
        self.X = X
        self.U = U
        self.A = np.identity(3)
        self.C = np.identity(3)
        self.x_km2_km1 = np.zeros(3)

        self.Sigma_init = np.array(
            [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.1]])  # <--------<< Initialize correction covariance
        self.sigma_measure = np.array([[0.05, 0, 0], [0, 0.05, 0],
                                       [0, 0, 0.1]])  # <--------<< Should be updated with variance from the measurement
        self.sigma_motion = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        self.KalGain = np.random.rand(3, 3)  # <--------<< Initialize Kalman Gain

        self.z_k = None
        self.z_height_k = None
        self.z_height_variance = None
        rospy.Subscriber(
            '/ballxyz/' + color,
            PoseWithCovarianceStamped,
            self.measurement_cb
        )

        self.Sx_k_k = self.Sigma_init

        self.belief_pub = rospy.Publisher('/ball_belief', PoseWithCovarianceStamped, queue_size=5)

    def prediction(self, x_km1_km1, Sigma_km1_km1):  ## Initial State is input
        
        x_mean_k_km1, x_mean_kpn_km1 = self.dotX(x_km1_km1) ## k_km1 becomes predicted step
        # TODO: display x_mean_kpn_km1 as future prediction

        # Defining A utilizing Jacobian... A = (I+Jacobian)
        Jacobian = [
            [x_km1_km1[0]-self.x_km2_km1[0], 0, 0],
            [0, x_km1_km1[1]-self.x_km2_km1[1], 0],
            [-1*np.sin(x_km1_km1[2]-self.x_km2_km1[2]),np.cos(x_km1_km1[2]-self.x_km2_km1[2]),x_km1_km1[2]-self.x_km2_km1[2]]
        ]
        Jacobian = np.divide(Jacobian, self.dt)
        self.A = np.add(np.identity(3),Jacobian)

        Sigma_k_km1 = np.matmul(self.A, Sigma_km1_km1, self.A.T) + self.sigma_motion

        self.x_km2_km1 = x_km1_km1

        return x_mean_k_km1, Sigma_k_km1

    def correction(self, x_mean_k_km1, Sx_k_km1, kalman_gain):
        x_mean_k_k = x_mean_k_km1 + np.matmul(kalman_gain, self.z_k)

        inner = np.identity(3) - np.matmul(kalman_gain, self.C)
        Sigma_k_k = np.matmul(inner, Sx_k_km1)

        return x_mean_k_k, Sigma_k_k

    def compute_gain(self, Sx_k_km1):
        inner = np.matmul(self.C, Sx_k_km1, self.C.T) + self.sigma_measure
        return np.matmul(Sx_k_km1, self.C.T, np.linalg.inv(inner))

    def update(self):
        if self.z_k is None:
            return

        x_mean_k_km1, Sx_k_km1 = self.prediction(self.X, self.Sx_k_k)
        kalman_gain = self.compute_gain(Sx_k_km1)
        self.X, self.Sx_k_k = self.correction(x_mean_k_km1, Sx_k_km1, kalman_gain)
        
        print(self.X, self.Sx_k_k)
        self.publish_ball_belief()

    def dotX(self, x):
        x_dot = np.asarray([
            self.U[0] * np.cos(x[2]) * self.dt,
            self.U[0] * np.sin(x[2]) * self.dt,
            self.U[1] * self.dt,
        ])

        return (x + x_dot), (x + self.nk * x_dot)

    def measurement_cb(self, pwcs):
        if self.z_k is None:
            theta = 0
        else:
            theta = np.arctan2(pwcs.pose.pose.position.y - self.z_k[1], pwcs.pose.pose.position.x - self.z_k[0])
        
        self.z_k = np.asarray([pwcs.pose.pose.position.x, pwcs.pose.pose.position.y, theta])

        self.sigma_measure = np.asarray([
            [pwcs.pose.covariance[0], 0, 0],
            [0, pwcs.pose.covariance[7], 0],
            [0, 0, 0.1],
        ])

        self.z_height_k = pwcs.pose.pose.position.z
        self.z_height_variance = min(pwcs.pose.covariance[14], 1)

    def publish_ball_belief(self):
        pwcs = PoseWithCovarianceStamped()
        pwcs.header.stamp = rospy.get_rostime()
        pwcs.header.frame_id = 'camera_link'

        x, y, z, w = quaternion_from_euler(0, 0, self.X[2])
        pwcs.pose.pose.orientation.x = x
        pwcs.pose.pose.orientation.y = y
        pwcs.pose.pose.orientation.z = z
        pwcs.pose.pose.orientation.w = w

        pwcs.pose.pose.position.x = self.X[0]
        pwcs.pose.pose.position.y = self.X[1]
        pwcs.pose.pose.position.z = self.z_height_k

        pwcs.pose.covariance = [
            self.Sx_k_k[0, 0], 0, 0, 0, 0, 0,
            0, self.Sx_k_k[1, 1], 0, 0, 0, 0,
            0, 0, self.z_height_variance, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ]

        self.belief_pub.publish(pwcs)


if __name__ == '__main__':
    rospy.init_node('EKF')

    # ---------------Define initial conditions --------------- #
    dt = 1  # Sampling duration (seconds)
    nk = 1  # Look ahead duration (seconds)

    # Initial State of the ball
    X = [0, 0, 0]

    # Control input, always assumed to be going straight at constant velocity
    U = [
        0.5,  # Forward velocity (meters / second)
        0  # Turning velocity (radians / second)
    ]

    extended_kalman_filter = EKF(nk, dt, X, U)

    hz = 1.0 / dt
    rate = rospy.Rate(hz)
    while not rospy.is_shutdown():
        extended_kalman_filter.update()
        rate.sleep()

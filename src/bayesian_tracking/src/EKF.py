#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import quaternion_from_euler


class EKF():
    def __init__(self, nk, dt, X, U, color='Purple'):
        self.nk = nk
        self.dt = dt
        self.X = X
        #self.X_predicted_steps = zeros(nk)
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
        self.future_pub = rospy.Publisher('/future', MarkerArray, queue_size=5)

    def prediction(self, x_km1_km1, Sigma_km1_km1):

        # ADDED THIS BIT _______________
        # Calculate nk time steps ahead
        x_predictions = []
        for time_steps in range(self.nk):
        
            only_dot = True
            if only_dot:
        
                x_mean_kn_km1 = self.dotX(x_km1_km1, time_steps + 1)
            else:
            
                x_mean_kn_km1 = self.dotX(
                    np.matmul(np.linalg.matrix_power(self.A, time_steps + 1), x_km1_km1), 
                    time_steps + 1
                )
            
            x_predictions.append(x_mean_kn_km1)
        #_______________________________
        # Defining A utilizing Jacobian... A = (I+Jacobian)
        """
        jacobian = np.asarray([
            [x_km1_km1[0]-self.x_km2_km1[0], 0, 0],
            [0, x_km1_km1[1]-self.x_km2_km1[1], 0],
            [-1*np.sin(x_km1_km1[2]-self.x_km2_km1[2]),np.cos(x_km1_km1[2]-self.x_km2_km1[2]),x_km1_km1[2]-self.x_km2_km1[2]]
        ])
        """
        x_mean_k_km1 = np.matmul(self.A, x_km1_km1)
        Jacobian = np.asarray([
            [x_mean_k_km1[0] - x_km1_km1[0], 0, 0],
            [0, x_mean_k_km1[1] - x_km1_km1[1], 0],
            [-1*np.sin(x_mean_k_km1[2]-x_km1_km1[2]),np.cos(x_mean_k_km1[2]-x_km1_km1[2]),x_mean_k_km1[2] - x_km1_km1[2]]
        ])
        
        
        """
        Jacobian = np.asarray([
            [x_mean_k_km1[0] - x_km1_km1[0], 0, 0],
            [0, x_mean_k_km1[1] - x_km1_km1[1], 0],
            [0, 0,x_mean_k_km1[2] - x_km1_km1[2]]
        ])
        """
        """
        Jacobian = np.asarray([
            [x_km1_km1[0] - self.x_km2_km1[0], 0, 0],
            [0, x_km1_km1[1] - self.x_km2_km1[1], 0],
            [0, 0,x_km1_km1[2] - self.x_km2_km1[2]]
        ])
        """
        
        Jacobian = np.divide(Jacobian, self.dt)
        self.A = np.add(np.identity(3), Jacobian)
        
        # print(self.A, self.dt)
        
        #x_mean_k_km1 = np.matmul(self.A, x_km1_km1)

        Sigma_k_km1 = np.matmul(np.matmul(self.A, Sigma_km1_km1), self.A.T) + self.sigma_motion

        self.x_km2_km1 = x_km1_km1

        return x_predictions, x_mean_k_km1, Sigma_k_km1

    def correction(self, x_mean_k_km1, Sx_k_km1, kalman_gain):
        x_mean_k_k = x_mean_k_km1 + np.matmul(kalman_gain, self.z_k - x_mean_k_km1)

        inner = np.identity(3) - np.matmul(kalman_gain, self.C)
        Sigma_k_k = np.matmul(inner, Sx_k_km1)

        return x_mean_k_k, Sigma_k_k

    def compute_gain(self, Sx_k_km1):
        inner = np.matmul(np.matmul(self.C, Sx_k_km1), self.C.T) + self.sigma_measure
        return np.matmul(np.matmul(Sx_k_km1, self.C.T), np.linalg.inv(inner))

    def update(self):
        if self.z_k is None:
            return
        
        # Output matrix of means of all predicted steps
        x_prediction_means, x_mean_k_km1, Sx_k_km1 = self.prediction(self.X, self.Sx_k_k)
        #x_mean_k_km1 = x_prediction_means[0] #Changed this line too
        kalman_gain = self.compute_gain(Sx_k_km1)
        self.X, self.Sx_k_k = self.correction(x_mean_k_km1, Sx_k_km1, kalman_gain)
        """for i in range(nk):
            self.X_predicted_steps[i],Sx_k_k_step = self.correction(s_prediction_means[i], Sx_k_km1, kalman_gain)"""
        #print(self.X, self.Sx_k_k)
        
        xs, ys = [], []
        for x, y, z in x_prediction_means:
            xs.append(x)
            ys.append(y)
        print('Belief Pose => ', self.X)
        print('Belief Covariance => ', self.Sx_k_k)
        self.publish_future(xs, ys, self.z_height_k)

        self.publish_ball_belief()
    
    def dotX(self, x, time_step):
        x_dot = np.asarray([
            self.U[0] * np.cos(x[2]) * self.dt,
            self.U[0] * np.sin(x[2]) * self.dt,
            self.U[1] * self.dt,
        ])

        #return (x + x_dot), (x + self.nk * x_dot)
        return (x + time_step * x_dot)

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
    
    def publish_future(self, xs, ys, z, diameter=0.15):

        markers = MarkerArray()
        markers.markers = []
        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            
            marker = Marker()
            marker.header.frame_id = '/camera_link'
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.ns = 'FutureBalls'
            marker.id = i
            marker.scale.x = diameter
            marker.scale.y = diameter
            marker.scale.z = diameter
            marker.color.a = 0.5 + (len(xs) - i) * 0.5
            marker.color.r = 0
            marker.color.g = 1
            marker.color.b = 0
            marker.pose.orientation.w = 1
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            
            markers.markers.append(marker)

        self.future_pub.publish(markers)


if __name__ == '__main__':
    rospy.init_node('EKF')

    # ---------------Define initial conditions --------------- #
    dt = .8  # Sampling duration (seconds)
    nk = 5  # Look ahead duration (seconds)

    # Initial State of the ball
    X = [0, 0, 0]

    # Control input, always assumed to be going straight at constant velocity
    U = [
        0.17,  # Forward velocity (meters / second)
        0  # Turning velocity (radians / second)
    ]

    extended_kalman_filter = EKF(nk, dt, X, U)

    hz = 1.0 / dt
    rate = rospy.Rate(hz)
    while not rospy.is_shutdown():
        extended_kalman_filter.update()
        rate.sleep()

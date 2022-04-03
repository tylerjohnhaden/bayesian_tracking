import numpy as np
import matplotlib.pyplot as plt
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped

class EKF():
    def __init__(self, nk, dt, X, U):
        self.nk = nk
        self.dt = dt
        self.X = X
        self.U = U

        self.Sigma_init = np.array([[0.05, 0], [0, 0.05]])     # <--------<< Initialize corection covariance
        self.sigma_measure = np.array([[0.05, 0], [0, 0.05]])  # <--------<< Should be updated with variance from the measurement
        self.KalGain = np.random.rand(2, 2)                  # <--------<< Initialize Kalman Gain
    
        self.color = 'Purple'
        rospy.Subscriber(
            '/ballxyz/' + self.color,
            PoseWithCovarianceStamped,
            self.measurement_cb
        print(self.z_k)

        self.Sx_k_k = self.Sigma_init



    def prediction(self, x, U, Sigma_km1_km1):
        #TODO 
        # Use the motion model and input sequence to predict a n step look ahead trajectory. 
        # You will use only the first state of the sequence for the rest of the filtering process.
        # So at a time k, you should have a list, X = [xk, xk_1, xk_2, xk_3, ..., xk_n] and you will use only xk. 
        # The next time this function is called a lnew list is formed. 

    def correction(self,x_predict, Sx_k_km1, z_k, KalGain):
        #TODO
        # Write a function to correct your prediction using the observed state.


    def update(self):
        self.X_pred = self.X 
        
        X_predicted,Sx_k_km1, A = self.prediction(self.X,self.U,self.Sx_k_k)                        # PREDICTION STEP  
        X_corrected, self.Sx_k_k = self.correction(X_predicted, Sx_k_km1, self.z_k, self.KalGain)   # CORRECTION STEP 
        self.gainUpdate(Sx_k_km1)                                                                   # GAIN UPDATE       
        self.X = X_corrected  

        self.X_pred = np.reshape(self.X_pred, [6, 2])       
        self.X_correc = np.reshape(self.X_correc, [6, 2])   # <--------<< Publish 

        self.X = self.X_correc

    def gainUpdate(self,Sx_k_km1):
        #TODO
        # Write a function to update the Kalman Gain, a.k.a. self.KalGain

    
    def dotX(self,x,U):
        # TODO 
        # This is where your motion model should go. The differential equation.
        # This function must be called in the self.predict function to predict the future states.

    def getGrad(self,x,U):
        # TODO
        # Linearize the motion model here. It should be called in the self.predict function and should yield the A and B matrix.

        


if __name__ == '__main__':
    rospy.init_node('EKF')

    # ---------------Define initial conditions --------------- #
    nk = None       # <------<< Look ahead duration in seconds
    dt = 1          # <------<< Sampling duration of discrete model
    X =  [0, 0, 0]  # <------<< Initial State of the Ball
    U =  None       # <------<< Initial input to the motion model
    
    extended_kalman_filter = EKF(nk, dt, X, U)

    rate = rospy.Rate(dt)
    while not rospy.is_shutdown():
    
        extended_kalman_filter.update()
        rate.sleep()
 

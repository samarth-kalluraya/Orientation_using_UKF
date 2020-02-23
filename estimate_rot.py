"""
4 DOF

"""

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter
import scipy
from scipy import io
import os
import numpy as np
from quaternions import *
from four_ukf import *
import matplotlib.pyplot as plt


def convert_data(z,data_num):
    acc=z[:3,:]
    gyro_x=z[4,:]
    gyro_y=z[5,:]
    gyro_z=z[3,:]
    gyro=np.array([gyro_x,gyro_y,gyro_z])
       
    acc_sens = 33.6
    acc_sf = 3300/(1023*acc_sens)
    acc_bias = np.array([[511],[501],[503.82]])
    if data_num==1:
        acc_bias = np.array([[511],[501],[502.7]])
    acc_real = (acc-acc_bias)*acc_sf
    
    gyro_s = 3.3
    gyro_sf = 3300*np.pi/(1023*180*gyro_s)  
    gyro_bias = np.mean(gyro[:250], axis = 1).reshape(3,1)
    gyro_real = (gyro-gyro_bias)*gyro_sf
    
    z_alog = np.zeros(z.shape)
    
    z_alog[:3,:] = acc_real
    z_alog[3:,:] = gyro_real
    
    z_alog[0:2,:] = -z_alog[0:2,:]

    return z_alog

def rotationMatrixToEulerAngles(Rot) :

    theta = -np.arcsin(Rot[2,0,:])
    psi = np.arctan2(Rot[2,1,:]/np.cos(theta), Rot[2,2,:]/np.cos(theta))
    phi = np.arctan2(Rot[1,0]/np.cos(theta), Rot[0,0]/np.cos(theta))
    return theta, psi, phi

##########      estimate function      ##########     
def estimate_rot(data_num=2):
 	#code goes here
    
    filename = os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(data_num) + ".mat")
    imuRaw = io.loadmat(filename)   
    imu_measurements = np.asarray(imuRaw['vals'])
    imu_ts =  np.asarray(imuRaw['ts'].T)
    imu_data=convert_data(imu_measurements,data_num)
    
    sigma_param=0.9*3
    x_hat = np.array([[1], [0], [0], [0]])
    P = 0.000000001 * np.identity(3) 
    
    roll = np.zeros(imu_ts.shape[0])
    pitch = np.zeros(imu_ts.shape[0])
    yaw =np.zeros(imu_ts.shape[0])
    
    if data_num==1: 
        Q = np.array([  0.00008,    0.00008,    0.00008])* np.identity(3)
        R = np.array([  0.005,     0.005,     0.005])* np.identity(3)  
        sigma_param=0.5*3
        imu_data[3:,:5]=0
        
    if data_num in (2,4,5,6):
        Q = np.array([  0.000001,    0.000001,    0.000001])* np.identity(3) 
        R = np.array([  0.004,     0.004,     0.004])* np.identity(3) 
        sigma_param=0.5*3
        imu_data[3:,:5]=0
    
    if data_num==3:
        sigma_param=0.1*3
        Q = np.array([  0.00005,    0.00002,    0.002])* np.identity(3) 
        R = np.array([  0.01,    0.01,    0.01])* np.identity(3) 
        imu_data[3:,:500]=0
        x_hat = np.array([[0.9990466], [-0.0098041], [-0.0076508], [-0.0418467]])

    dt=imu_ts[1]-imu_ts[0]
    
    for i in range(imu_ts.shape[0]):
        Xi=generate_sigma_pts(P,Q,x_hat,sigma_param)
        Yi= process_model(Xi,imu_data[:,i],dt)  
        Yi_mean, error_vecs = get_mean_state(Yi)
        P_dash,W_i_dash = get_covariance(Yi,Yi_mean,error_vecs)
        Zi,z_mean = get_measurement_model_and_mean(Yi)
        K, K_Vk_quat, Pvv = get_kalman(Zi,z_mean,imu_data[:,i].reshape(6,1),R,W_i_dash)
        x_hat,P =  measurement_update(Yi_mean,P_dash,K, K_Vk_quat, Pvv)
        dt=imu_ts[i]-imu_ts[i-1]
        roll[i], pitch[i], yaw[i]=q2rpy(x_hat) 

    return roll,pitch,yaw


##########  run main 

if __name__ == "__main__":
    
    data_num = 3
    
    roll,pitch,yaw=estimate_rot(data_num)
   
    filename = os.path.join(os.path.dirname(__file__), "vicon/viconRot" + str(data_num) + ".mat")
    viconData = io.loadmat(filename)
    
    theta, psi, phi = rotationMatrixToEulerAngles(viconData['rots'])
    
    p = plt.figure(1)
    ax = p.add_subplot(111)
    ax.plot(psi, label="vicon")
    ax.plot(roll, label="estimated") 
    ratio = 0.4
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    plt.title('Roll')
    plt.legend(loc="upper left") 
    plt.show()
    
    q = plt.figure(1)
    ax = q.add_subplot(111)
    ax.plot(theta, label="vicon")
    ax.plot(pitch, label="estimated") 
    ratio = 0.4
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    plt.title('Pitch')
    plt.legend(loc="upper left")  
    plt.show()

    r = plt.figure(1)
    ax = r.add_subplot(111)
    ax.plot(phi, label="vicon")
    ax.plot(yaw, label="estimated") 
    ratio = 0.4
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    plt.title('Yaw')
    plt.legend(loc="upper left")

    plt.show()

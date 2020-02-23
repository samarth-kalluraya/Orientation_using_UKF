#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:26:47 2020

@author: samarth
"""

import numpy as np
from quaternions import *

g=np.array([[0],[0],[0],[9.8]])

def generate_sigma_pts(P,Q,x_hat,sigma_param):
    S = np.linalg.cholesky(P+Q)
    S = S*np.sqrt(sigma_param)
    W=np.hstack((S,-S))
    quats=vec2q(W[:3,:])
    W = quats
    Xi=np.zeros_like(W)
    Xi = q_multiply(x_hat,W)
    return Xi
#       Xi=generate_sigma_pts(P,Q,x_hat)

def process_model(Xi,z_imu,dt):
    w=z_imu.copy()
    w=w[3:].reshape(3,1)
    w_norm=np.linalg.norm(w)
    alfa_d=w_norm*dt
    if w_norm==0:
        e_d=w
    else:
        e_d= w/w_norm
    quat_d=np.zeros((4,1))
    quat_d[0]=np.cos(alfa_d/2)
    quat_d[1:]=np.sin(alfa_d/2)*e_d
    Yi=np.zeros_like(Xi)
    Yi = q_multiply(Xi,quat_d)
    return Yi
#       Yi= process_model(Xi,x_hat,dt)      
    
def get_mean_state(Yi):
    Yi_mean=np.zeros((4,1))
    Yi_mean,error_vecs=q_avg(Yi)
    return Yi_mean, error_vecs
#       Yi_mean, error_vecs = get_mean_state(Yi)


def get_covariance(Yi,Yi_mean,error_vecs):
    W_i_dash=np.zeros((3,Yi.shape[1]))
    W_i_dash=error_vecs
    P_dash=np.zeros((3,3))
    P_dash=P_dash+np.dot(W_i_dash,W_i_dash.T)
    P_dash=P_dash/6
    return P_dash,W_i_dash
#       P_dash,W_i_dash = get_covariance(Yi,Yi_mean,error_vecs)


def get_measurement_model_and_mean(Yi):
    Zi=np.zeros((3,6))
    a_quat=np.zeros((4,6))
    a_quat=q_multiply(q_inverse(Yi), g)
    a_quat=q_multiply_1on1(a_quat, Yi)    
    Zi=q2vec(a_quat)
    z_mean=np.zeros((3,1))
    z_mean=np.mean(Zi,axis=1).reshape(3,1)
    return Zi,z_mean
#   Zi,z_mean = get_measurement_model_and_mean(Yi)
    
def get_kalman(Zi,z_mean,z_imu,R,W_i_dash):
    Vk=z_imu[:3]-z_mean
    
    Pzz=np.zeros((3,3))
    temp=Zi-z_mean
    Pzz=Pzz+np.dot(temp,temp.T)
    Pzz=Pzz/6
    
    Pvv=Pzz+R
    
    Pxz=np.zeros((3,3))
    Pxz=Pxz+np.dot(W_i_dash,temp.T)
    Pxz=Pxz/6
    
    K = np.dot(Pxz, np.linalg.inv(Pvv))
    
    K_Vk_vec = np.dot(K,Vk)
    K_Vk_quat= np.zeros((4,1))
    K_Vk_quat= vec2q(K_Vk_vec)
    return K, K_Vk_quat, Pvv
#    K, K_Vk_quat, Pvv = get_kalman(Zi,z_mean,z_imu,R,W_i_dash)
    

def measurement_update(Yi_mean,P_dash,K, K_Vk_quat, Pvv):
    new_x = np.zeros((4,1))
    new_x=q_multiply(Yi_mean,K_Vk_quat) 
    new_P = P_dash - np.dot(np.dot(K,Pvv),K.T)
    return new_x,new_P
#   new_x,new_P =  measurement_update(Yi_mean,P_dash,K, K_Vk_quat, Pvv)
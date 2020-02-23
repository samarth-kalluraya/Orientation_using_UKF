#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 16:07:51 2020

@author: samarth
"""
import numpy as np

#tuning parameters
gd_iter=1000
error_thresh=0.0001    #used in q_avg

def q_conjugate(q):  #can be 4xm or 4x1
    ans=q.copy()
    ans[1:,:]=-ans[1:,:]
    return ans

def q_normalize(q): #can be 4xm or 4x1
    temp=q.copy()
    norm=np.linalg.norm(temp,axis=0)
    return (temp/norm)
    

def q_inverse(q):  #can be 4xm or 4x1
    temp=q_conjugate(q)
    norm=np.linalg.norm(temp,axis=0)
    return (temp/norm**2)
    
def q_multiply(q1,q2):      
    if q1.shape[1]==1:
        ans=np.zeros_like(q2)
        ans[0,:] = q1[0]*q2[0,:]-np.dot(q1[1:,:].T, q2[1:,:])
        ans[1:,:] = q1[0]*q2[1:,:]+q2[0,:]*q1[1:] + np.cross(q1[1:],q2[1:,:],axis=0)
        ans=q_normalize(ans)
        
    if q2.shape[1]==1:
        ans=np.zeros_like(q1)
        ans[0,:] = q1[0,:]*q2[0]-np.dot(q2[1:,:].T, q1[1:,:])
        ans[1:,:] = q2[0]*q1[1:,:]+q1[0,:]*q2[1:] - np.cross(q2[1:],q1[1:,:],axis=0)
        ans=q_normalize(ans)
    return ans

def q_multiply_1on1(q1,q2):      #both q1 and q2 4xm
    ans=np.zeros_like(q1)
    ans[0,:] = q1[0,:]*q2[0,:]-np.sum(np.multiply(q1[1:],q2[1:]),axis=0)
    ans[1:,:] = q1[0,:]*q2[1:,:]+q2[0,:]*q1[1:,:] + np.cross(q1[1:,:],q2[1:,:],axis=0)
    ans=q_normalize(ans)
        
    return ans


def q2rotmat(q):  #2 D array   *** normalized
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    R = np.array([[1-2*y**2-2*z**2, 2*x*y-2*w*z, 2*x*z+2*w*y],
                  [2*x*y+2*w*z, 1-2*x**2-2*z**2, 2*y*z-2*w*x],
                  [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x**2-2*y**2]])
    return R

def q2rpy(q):    #2 D array   *** normalized
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    roll =np.arctan2(2*w*x+2*y*z , 1-2*x**2-2*y**2)
    pitch=np.arcsin(2*w*y-2*x*z)
    yaw  =np.arctan2(2*w*z+2*x*y , 1-2*y**2-2*z**2)
    return roll,pitch,yaw
    
def q2vec(q):   #can be 4xm or 4x1
    temp=q_normalize(q)
    theta_by2=np.arccos(temp[0,:])
    theta_by2[theta_by2==0]=0.000001
    sin_theta_by2=np.sin(theta_by2)
    ew=temp[1:,:]/sin_theta_by2
    ew=ew*2*theta_by2
    return ew

def vec2q(v):   #can be 3xm or 3x1
    temp = v.copy()
    theta= np.linalg.norm(temp,axis=0)
    theta[theta==0]=0.000001
    ew = temp/theta
    ans = np.zeros((temp.shape[0]+1,temp.shape[1]))
    sin_theta_by2=np.sin(theta/2)
    ans[0,:]=np.cos(theta/2)
    ans[1:,:]=ew*sin_theta_by2
    return ans

def q_avg(q):
    mean_q = q[:, 0].copy()
    mean_q = mean_q.reshape(4,1)
    q_internal = q.copy()
    m=q_internal.shape[1]
    error_quats=np.zeros((4,m))
    iterations=0
    while(iterations<gd_iter):
        mean_q_inv=q_inverse(mean_q)    #4x1
        error_quats=q_multiply(q_internal, mean_q_inv)
        error_vecs=q2vec(error_quats)
        mean_error_vec=np.mean(error_vecs,axis=1).reshape(3,1)
        if np.linalg.norm(mean_error_vec)<error_thresh:
            break
        mean_error_quat=vec2q(mean_error_vec)#.reshape(4)
        mean_q=q_multiply(mean_error_quat, mean_q).reshape(4,1)
        iterations+=1
    return q_normalize(mean_q), error_vecs


#can use slerp to test performance of q_avg
def slerp(v0, v1, t_array):
    """Spherical linear interpolation."""
    # >>> slerp([1,0,0,0], [0,0,0,1], np.arange(0, 1, 0.001))
    t_array = np.array(t_array)
    v0 = np.array(v0)
    v1 = np.array(v1)
    dot = np.sum(v0 * v1)

    if dot < 0.0:
        v1 = -v1
        dot = -dot
    
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = v0[np.newaxis,:] + t_array[:,np.newaxis] * (v1 - v0)[np.newaxis,:]
        return (result.T / np.linalg.norm(result, axis=1)).T
    
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t_array
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:,np.newaxis] * v0[np.newaxis,:]) + (s1[:,np.newaxis] * v1[np.newaxis,:])
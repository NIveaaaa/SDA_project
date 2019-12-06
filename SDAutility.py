# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:07:10 2019

@author: z5187692
"""

import numpy as np
from scipy.special import comb

def getX_order1(derivative):
    Z = -0.5*derivative
    return(Z) 
    
    
def getX_order2(sample,derivative):
    Z = -0.5*derivative
    theta_dim = sample.shape[1]
    squared = 2*np.multiply(sample,Z)-1
    twoway = np.zeros(shape=(sample.shape[0], int(comb(theta_dim,2))))
    pos = -1
    for j in range(theta_dim-1):
        for ii in range(j+1,theta_dim,1):
            pos = pos+1
            twoway[:,pos] = np.multiply(sample[:,ii],Z[:,j])+np.multiply(sample[:,j],Z[:,ii])
    poly2 =  np.concatenate([Z,squared,twoway],axis=1)
    return(poly2)


def compute_integral_1st(est_expect,t_space):
    end_point = est_expect
    left_point = end_point[0:-1]
    right_point = end_point[1:]
    final_result = np.multiply(np.diff(t_space),np.mean([left_point,right_point],axis=0))
    return end_point,np.sum(final_result)+end_point[0]*t_space[0]



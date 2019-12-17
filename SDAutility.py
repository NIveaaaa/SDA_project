# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:07:10 2019

@author: z5187692
"""

import numpy as np
from scipy.special import comb
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects

TruncatedNormal = rpackages.importr('TruncatedNormal')

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
    keep_index = np.isfinite(est_expect)
    end_point = est_expect[keep_index]
    t_space_temp = t_space[keep_index]
    left_point = end_point[0:-1]
    right_point = end_point[1:]
    final_result = np.multiply(np.diff(t_space_temp),
                               np.mean([left_point,right_point],axis=0))
    return end_point,np.sum(final_result)+end_point[0]*t_space[0]

def Rrtmvnorm(mu,sigma,smin,smax,nsample):
    try:
        TruncatedNormal
    except Error:
        print('no moulde loaded')
    else:
        sigma0 = np.array(sigma).flatten()
        mu0 = np.array(mu)
        smin0 = np.array(smin)
        smax0 = np.array(smax)
        # convert to r matrix
        sigma_r = robjects.r['matrix'](robjects.FloatVector(sigma0),nrow=mu0.shape[0])
        mu_r =robjects.FloatVector(mu0)
        a = robjects.FloatVector(smin0)
        b = robjects.FloatVector(smax0)
        x = TruncatedNormal.rtmvnorm(nsample,mu_r,sigma_r,a,b)
        return np.asarray(x)


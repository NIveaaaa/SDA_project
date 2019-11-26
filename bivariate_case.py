# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:39:22 2019

@author: z5187692
"""

import numpy as np
import math
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import kstest
import scipy.integrate as intergrate
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import multivariate_normal
import SDAutility as Sutil
#%%
class Bivariate_Normal_Path_sample:
    
    def __init__(self,smin,smax,t,mu,sigma,\
                 nsample = None, use_CV = False, CV_order = 0):
        self.smin = smin
        self.smax = smax
        self.t = t
        self.mu = mu
        self.sigma =sigma
        self.sigma_at_t = np.divide(self.sigma,self.t)
        if nsample is None:
            self.nsample = int(1000)
        else: self.nsample = int(nsample)
        self.z = np.zeros([self.nsample,2])
        self.use_CV = use_CV 
        self.CV_order = CV_order
        self.phi_theta = np.zeros(self.nsample)

    
        # use gibbs sampler to sample from a truncated bivariate normal dist
    def sample_z(self):
        z = np.zeros([self.nsample,2])
        sd0 = np.sqrt(self.sigma_at_t[0,0])
        sd1 = np.sqrt(self.sigma_at_t[1,1])
        corr = self.sigma_at_t[1,0]/(sd0 * sd1)
        smin_trans = np.divide(self.smin - self.mu, [sd0,sd1])
        smax_trans = np.divide(self.smax - self.mu, [sd0,sd1])
        for i in range(self.nsample-1):
            cmean_1 = corr*z[i,1]
            csd = np.sqrt(1 - np.power(corr,2))
            a_1 = (smin_trans[0] - cmean_1)/csd
            b_1 = (smax_trans[0] - cmean_1)/csd
            z[i+1,0] = truncnorm.rvs(a_1,b_1,loc=cmean_1,scale=csd)
            
            cmean_2 = corr*z[i+1,0]
            a_2 = (smin_trans[1] - cmean_2)/csd
            b_2 = (smax_trans[1] - cmean_2)/csd
            
            z[i+1,1] = truncnorm.rvs(a_2,b_2,loc=cmean_2,scale=csd)
        self.z[:,0] = z[:,0]*sd0 + self.mu[0]
        self.z[:,1] = z[:,1]*sd1 + self.mu[1]



    
    def get_phi_theta(self):
        self.sample_z()
        self.phi_theta = multivariate_normal.logpdf(self.z,self.mu,self.sigma)
    
    def get_expectation(self):
        self.get_phi_theta()
        if self.use_CV:
            self.get_cv()
            self.expectation = np.mean(self.phi_theta-self.h_theta)
        else:
            self.expectation = np.mean(self.phi_theta)
    
    def get_cv(self):
        # get a n by 2 derivative function
        self.deriv = -1*np.dot(np.linalg.inv(self.sigma_at_t),(self.z-self.mu).T).T

        # order 1
        if (self.CV_order==1):
            x = Sutil.getX_order1(self.deriv)
 
        # order 2
        if (self.CV_order==2):
            x = Sutil.getX_order2(self.z.reshape((self.nsample,2)),self.deriv.reshape((self.nsample,2)))
  
        x_theta = sm.add_constant(x)
        model = sm.OLS(self.phi_theta, x_theta)
        results = model.fit()
        self.h_theta = np.sum( np.multiply(x,results.params[1:]),axis=1)

            
        

def log_phi1(smin,smax,mu,sigma):
    comp1 = multivariate_normal.cdf(smax,mu,sigma)
    comp2 = multivariate_normal.cdf(smin,mu,sigma)
    comp3 = multivariate_normal.cdf([smin[0],smax[1]],mu,sigma)
    comp4 = multivariate_normal.cdf([smin[1],smax[0]],mu,sigma)
    return np.log(comp1+comp2-comp3-comp4)
    



def compute_integral_2nd(smin,smax,mu,sigma,t_space,nsample=1000,use_CV=False,CV_order=0):
    mean_end_point = np.zeros(t_space.shape[0])
    var_end_point = np.zeros(t_space.shape[0])
    for i in np.arange(start=0,stop=t_space.shape[0],step=1):
        uni = Bivariate_Normal_Path_sample(smin,smax,t_space[i],mu,sigma,\
                                           nsample,use_CV,CV_order)
        uni.get_expectation()
        mean_end_point[i] = uni.expectation
        var_end_point[i] = np.std(uni.z)
    diff_t_space = np.diff(t_space)
    first_approx = np.multiply(diff_t_space,\
                              np.mean([mean_end_point[0:-1],\
                                       mean_end_point[1:]],axis=0))
    quad_approx = first_approx - np.multiply(np.power(diff_t_space,2),\
                         np.diff(var_end_point))/12
    return(np.sum(first_approx),np.sum(quad_approx))
    
                                             
#%%    

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:39:22 2019

@author: z5187692
"""

import numpy as np
import math
from scipy.stats import truncnorm
from scipy.stats import norm
import scipy.integrate as intergrate
import matplotlib.pyplot as plt

class Univariate_Normal_Path_sample:
    
    def __init__(self,smin,smax,t,mu,sigma,nsample = None):
        self.smin = smin
        self.smax = smax
        self.t = t
        self.mu = mu
        self.sigma =sigma
        self.sigma_at_t = self.sigma/math.sqrt(self.t)
        if nsample is None:
            self.nsample = 1000
        else: self.nsample = nsample
        self.expectation = None
        self.z = None
         
    def expectation_z(self):
        a = (smin - self.mu)/self.sigma_at_t
        b = (smax - self.mu)/self.sigma_at_t
        self.z = truncnorm.rvs(a,b,loc=self.mu,\
                               scale = self.sigma_at_t,\
                               size=self.nsample)
        trans_z = (self.z - self.mu)/self.sigma
        self.expectation = -0.5*np.log(2*np.pi)-np.log(self.sigma)-\
        0.5*np.mean(np.power(trans_z,2))
        #self.expectation = np.mean(norm.logpdf(self.z,loc=self.mu,scale=self.sigma))


def intergrand(smin,smax,t,mu,sigma,nsample=100):
    uni = Univariate_Normal_Path_sample(smin,smax,t,mu,sigma)
    uni.expectation_z()
    return(uni.expectation)


def log_phi1(smin,smax,mu,sigma):
    return(np.log(norm.cdf(smax,loc=mu,scale=sigma)\
                    -norm.cdf(smin,loc=mu,scale=sigma)))
    

def compute_integral(smin,smax,mu,sigma,t_space):
    end_point = np.zeros(t_space.shape[0])
    for i in np.arange(start=0,stop=t_space.shape[0],step=1):
        uni = Univariate_Normal_Path_sample(smin,smax,t_space[i],mu,sigma)
        uni.expectation_z() # compute the expecation
        end_point[i] = uni.expectation
    left_point = end_point[0:-1]
    right_point = end_point[1:]
    final_result = np.multiply(np.diff(t_space),np.mean([left_point,right_point],axis=0))
    return(final_result,np.sum(final_result))
    
#%%    
        
smin = -2
smax = 2
mu = 0
sigma = 1


log_phi0 = np.log(smax-smin)


result = intergrate.quad(intergrand,0.00,1,args=(smin,smax,mu,sigma))
print('estimation of log phi1-[by intergrate.quad]:',result[0]+log_phi0)

# est_log_phi1 = log_phi0 + result[0]
true_log_phi1 = log_phi1(smin=smin,smax=smax,mu=mu,sigma=sigma)


t_space = np.log(np.logspace(start=0.0001,stop=1,base=np.e,num=2000))
#t_space = np.linspace(start=0.001,stop=1,num=2000)
est_log_phi1_per_t,est_log_phi1 = compute_integral(smin,smax,mu,sigma,t_space)

print('estimation of log phi1:', np.sum(est_log_phi1)+log_phi0)
print('true value of log phi1:', true_log_phi1)

plt.figure(figsize=(6,6))
plt.plot(t_space[0:-1],est_log_phi1_per_t)
plt.close()
#%% replication of 50 times

nrep = 50
est_log_phi1_array = np.zeros(nrep)

for i in range(nrep):
    _,est_log_phi1_array[i] = compute_integral(smin,smax,mu,sigma,t_space)

plt.figure(figsize=(6,6))
plt.plot(est_log_phi1_array+log_phi0,'-',label='estimated value')
plt.axhline(true_log_phi1, color='k', linestyle='dashed', linewidth=1,\
            label='true value')
plt.axhline(np.mean(est_log_phi1_array)+log_phi0,linestyle='dashed',\
            color='r',label='mean value')
plt.legend()
plt.close()
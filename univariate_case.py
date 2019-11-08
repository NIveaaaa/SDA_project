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
import statsmodels.api as sm

class Univariate_Normal_Path_sample:
    
    def __init__(self,smin,smax,t,mu,sigma,nsample = None, use_CV = False):
        self.smin = smin
        self.smax = smax
        self.t = t
        self.mu = mu
        self.sigma =sigma
        self.sigma_at_t = self.sigma/np.sqrt(self.t)
        if nsample is None:
            self.nsample = 1000
        else: self.nsample = nsample
        self.expectation = None
        self.z = None
        self.use_CV = use_CV    
        # sample z here
        a = (smin - self.mu)/self.sigma_at_t
        b = (smax - self.mu)/self.sigma_at_t
        
        self.z = truncnorm.rvs(a,b,loc=self.mu,\
                               scale = self.sigma_at_t,\
                               size=self.nsample)
        self.phi_theta = None
        self.h_theta = None
    
    def get_phi_theta(self):
        self.phi_theta = -0.5*np.log(2*np.pi)-np.log(self.sigma)-\
        0.5*np.power((self.z - self.mu)/self.sigma,2)
    
    def get_expection(self):
        self.get_phi_theta()
        if self.use_CV:
            self.get_cv()
            self.expectation = np.mean(self.h_theta+self.phi_theta)
        else:
            self.expectation = np.mean(self.phi_theta)
        #self.expectation = np.mean(norm.logpdf(self.z,loc=self.mu,scale=self.sigma))
    
    def get_cv(self):
        # order 1
        # theta = [mu, sigma_at_t]
        # h_theta is the control variates
        # phi_theta is the origin estimates
        grad_theta = np.zeros(shape = (self.nsample,1))
        grad_theta[:,0] = (self.mu - self.z)/np.power(self.sigma_at_t,2) 
          
      
        x_theta = sm.add_constant(grad_theta)
        model = sm.OLS(self.phi_theta, x_theta)
        results = model.fit()
        self.h_theta = np.sum(np.multiply(grad_theta,results.params[1:]),axis=1)

        

def intergrand(smin,smax,t,mu,sigma,nsample=100):
    uni = Univariate_Normal_Path_sample(smin,smax,t,mu,sigma)
    uni.get_expection()
    return(uni.expectation)


def log_phi1(smin,smax,mu,sigma):
    return(np.log(norm.cdf(smax,loc=mu,scale=sigma)\
                    -norm.cdf(smin,loc=mu,scale=sigma)))
    

def compute_integral_1st(smin,smax,mu,sigma,t_space,nsample=1000,use_CV=False):
    end_point = np.zeros(t_space.shape[0])
    for i in np.arange(start=0,stop=t_space.shape[0],step=1):
        uni = Univariate_Normal_Path_sample(smin,smax,t_space[i],mu,sigma,nsample,use_CV)
        uni.get_expection() # compute the expecation
        end_point[i] = uni.expectation
    left_point = end_point[0:-1]
    right_point = end_point[1:]
    final_result = np.multiply(np.diff(t_space),np.mean([left_point,right_point],axis=0))
    return(final_result)


def compute_integral_2nd(smin,smax,mu,sigma,t_space,nsample=1000,use_CV=False):
    mean_end_point = np.zeros(t_space.shape[0])
    var_end_point = np.zeros(t_space.shape[0])
    for i in np.arange(start=0,stop=t_space.shape[0],step=1):
        uni = Univariate_Normal_Path_sample(smin,smax,t_space[i],mu,sigma,nsample,use_CV)
        uni.get_expection()
        mean_end_point[i] = uni.expectation
        var_end_point[i] = np.std(uni.z)
    diff_t_space = np.diff(t_space)
    first_approx = np.multiply(diff_t_space,\
                              np.mean([mean_end_point[0:-1],\
                                       mean_end_point[1:]],axis=0))
    quad_approx = first_approx - np.multiply(np.power(diff_t_space,2),\
                         np.diff(var_end_point))/12
    return(first_approx,quad_approx)
                                             
#%%    
        
smin = -2
smax = 2
mu = 0
sigma = 1


log_phi0 = np.log(smax-smin)


#result = intergrate.quad(intergrand,0.00,1,args=(smin,smax,mu,sigma))
#print('estimation of log phi1-[by intergrate.quad]:',result[0]+log_phi0)

# est_log_phi1 = log_phi0 + result[0]
true_log_phi1 = log_phi1(smin=smin,smax=smax,mu=mu,sigma=sigma)


t_space = np.log(np.logspace(start=0.0001,stop=1,base=np.e,num=2000))
#t_space = np.linspace(start=0.001,stop=1,num=2000)
est_log_phi1_per_t = compute_integral_1st(smin,smax,mu,sigma,t_space)

print('estimation of log phi1:', np.sum(est_log_phi1_per_t)+log_phi0)
print('true value of log phi1:', true_log_phi1)

plt.figure(figsize=(6,6))
plt.plot(t_space[0:-1],est_log_phi1_per_t)

plt.savefig('true_vs_estimation.png')
plt.close()
#%% replication of 200 times

nrep = 100
est_log_phi1_1st_app = np.zeros(nrep)
est_log_phi1_2nd_app = np.zeros(nrep)
for i in range(nrep):
    first_app,second_app= compute_integral_2nd(smin,smax,mu,sigma,t_space)
    est_log_phi1_1st_app[i] = np.sum(first_app)
    est_log_phi1_2nd_app[i] = np.sum(second_app)

# almost no difference between 1st/2nd order approximation
plt.figure(figsize=(6,6))
plt.plot(est_log_phi1_1st_app+log_phi0,'-',color='orange')
plt.plot(est_log_phi1_2nd_app+log_phi0,'-.',color='k')
plt.axhline(true_log_phi1, color='k', linestyle='dashed', linewidth=1,\
            label='true value')
plt.axhline(np.mean(est_log_phi1_1st_app)+log_phi0,linestyle='dashed',\
            color='r',label='mean value 1st order')
plt.axhline(np.mean(est_log_phi1_2nd_app)+log_phi0,linestyle='dashed',\
            color='b',label='mean value 2nd order')
plt.legend()
plt.savefig('1st vs 2nd quadrature approximaiton.png')
plt.show()

plt.figure(figsize=(6,6))
plt.hist(est_log_phi1_1st_app+log_phi0)
plt.axvline(true_log_phi1, color='k', linestyle='dashed', linewidth=1,\
            label='true value')
plt.axvline(np.mean(est_log_phi1_1st_app)+log_phi0,linestyle='dashed',\
            color='r',label='mean value')
plt.title('100 replications')
plt.savefig('100 replication of log estimate.png')
plt.legend()

#%% Let's consider control variates

cv_result = compute_integral_1st(smin,smax,mu,sigma,t_space,1000,True)
 

plt.plot(t_space[0:-1],est_log_phi1_per_t)
plt.plot(t_space[0:-1],cv_result,color='k')
plt.show()
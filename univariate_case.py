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
#%%
class Univariate_Normal_Path_sample:
    
    def __init__(self,smin,smax,t,mu,sigma,\
                 nsample = None, use_CV = False, CV_order = 1):
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
        self.CV_order = CV_order
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
        if (self.CV_order==1):
            x = np.zeros(shape = (self.nsample,1))
            x[:,0] = (self.mu - self.z)/np.power(self.sigma_at_t,2) 
        if (self.CV_order==2):
            x = np.zeros(shape=(self.nsample ,2))
            x[:,0] = (self.mu - self.z)/np.power(self.sigma_at_t,2)
            x[:,1] = 2+ 2*np.multiply(x[:,0],self.z)
        x_theta = sm.add_constant(x)
        model = sm.OLS(self.phi_theta, x_theta)
        results = model.fit()
        self.h_theta = np.sum(np.multiply(x,results.params[1:]),axis=1)

        

def intergrand(smin,smax,t,mu,sigma,nsample=100):
    uni = Univariate_Normal_Path_sample(smin,smax,t,mu,sigma)
    uni.get_expection()
    return(uni.expectation)


def log_phi1(smin,smax,mu,sigma):
    return(np.log(norm.cdf(smax,loc=mu,scale=sigma)\
                    -norm.cdf(smin,loc=mu,scale=sigma)))
    

def compute_integral_1st(smin,smax,mu,sigma,t_space,nsample=1000,use_CV=False,CV_order = 1):
    end_point = np.zeros(t_space.shape[0])
    for i in np.arange(start=0,stop=t_space.shape[0],step=1):
        uni = Univariate_Normal_Path_sample(smin,smax,t_space[i],mu,sigma,nsample,use_CV, CV_order)
        uni.get_expection() # compute the expecation
        end_point[i] = uni.expectation
    left_point = end_point[0:-1]
    right_point = end_point[1:]
    final_result = np.multiply(np.diff(t_space),np.mean([left_point,right_point],axis=0))
    return(end_point,final_result)


def compute_integral_2nd(smin,smax,mu,sigma,t_space,nsample=1000,use_CV=False,CV_order=1):
    mean_end_point = np.zeros(t_space.shape[0])
    var_end_point = np.zeros(t_space.shape[0])
    for i in np.arange(start=0,stop=t_space.shape[0],step=1):
        uni = Univariate_Normal_Path_sample(smin,smax,t_space[i],mu,sigma,nsample,use_CV,CV_order)
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
    
def true_second_moment(smin,smax,mu,sigma,t):
    sigma0 = 1/math.sqrt(2*t)
    a = 1/math.sqrt(2*sigma)*(smin - mu)
    b = 1/math.sqrt(2*sigma)*(smax - mu)
    alpha = (a-mu)/sigma0
    beta = (b-mu)/sigma0
    phi_alpha = norm.pdf(alpha)
    phi_beta = norm.pdf(beta)
    Phi_alpha = norm.cdf(alpha)
    Phi_beta = norm.cdf(beta)
    Z = Phi_beta - Phi_alpha
    mean_value = mu + (phi_alpha - phi_beta)/Z*sigma0
    var_value = np.power(sigma0,2)*(1+(alpha*phi_alpha-beta*phi_beta)/Z\
                         + np.power((phi_alpha-phi_beta)/Z,2))
    second_m = var_value+np.power(mean_value,2)
    return(-second_m-0.5*np.log(2*np.pi)-np.log(sigma))
                                             
#%%    
        
smin = -2
smax = 2
mu = 0
sigma = 1


log_phi0 = np.log(smax-smin)


# est_log_phi1 = log_phi0 + result[0]
true_log_phi1 = log_phi1(smin=smin,smax=smax,mu=mu,sigma=sigma)

t_space = np.log(np.logspace(start=0.0001,stop=1,base=np.e,num=2000))
#t_space = np.linspace(start=0.001,stop=1,num=2000)
est_log_phi1_per_t,est_log_phi1 = compute_integral_1st(smin,smax,mu,sigma,t_space)

true_2nd_moment = [true_second_moment(smin,smax,mu,sigma,t) for t in t_space]


print('estimation of log phi1:', np.sum(est_log_phi1)+log_phi0)
print('true value of log phi1:', true_log_phi1)

f = plt.figure(figsize=(6,6))
plt.plot(t_space,est_log_phi1_per_t,label='estimation')
plt.plot(t_space,true_2nd_moment,label='true value')
plt.xlabel('t')
plt.ylabel('expectation')
plt.legend()
f.savefig('true_vs_estimation.pdf',bbox_iches = 'tight')
plt.close()
#%% replication of 1000 times

"""
nrep = 10000
est_log_phi1_1st_app = np.zeros(nrep)
est_log_phi1_2nd_app = np.zeros(nrep)
for i in range(nrep):
    first_app,second_app= compute_integral_2nd(smin,smax,mu,sigma,t_space)
    est_log_phi1_1st_app[i] = np.sum(first_app)
    est_log_phi1_2nd_app[i] = np.sum(second_app)
"""

est_log_phi1_1st_app = np.load('replications_10000_MC_1st_app.npy')
est_log_phi1_2nd_app = np.load('replications_10000_MC_2nd_app.npy')
# almost no difference between 1st/2nd order approximation

f0 = plt.figure(figsize=(6,6))
plt.hist(est_log_phi1_1st_app-est_log_phi1_2nd_app)
plt.title('difference between 1st and 2nd order approximation')
f0.savefig('diff_between_1st_2nd_app.pdf',bbox_inches = 'tight')
plt.show()

f1 = plt.figure(figsize=(6,6))
plt.hist(est_log_phi1_1st_app+log_phi0)
plt.axvline(true_log_phi1, color='k', linestyle='dashed', linewidth=1,\
            label='true value')
plt.axvline(np.mean(est_log_phi1_1st_app)+log_phi0,linestyle='dashed',\
            color='r',label='mean value')
plt.title('10,000 replications')
f1.savefig('10,000_replication_of_log_estimate.pdf',bbox_inches='tight')
plt.legend()

f2 = plt.figure(figsize=(6,6))
f2 = sm.qqplot(est_log_phi1_1st_app, line='s')
plt.title('qq-plot for 10,000 replications')
f2.savefig('qq_plot_10,000_replications.pdf',bbox_inches='tight',dpi=100)
plt.show

kstest(est_log_phi1_1st_app, 'norm',args = (np.mean(est_log_phi1_1st_app),\
                                            np.std(est_log_phi1_1st_app)))
#%% Let's consider control variates

# 1st order approximation
cv_1st,cv_1st_grid = compute_integral_1st(smin,smax,mu,sigma,t_space,1000,True,1)

# 2nd order approximation does not work
# cv_2nd,_, = compute_integral_1st(smin,smax,mu,sigma,t_space,1000,True,2)

f3 = plt.figure(figsize=(6,6))
plt.plot(t_space,est_log_phi1_per_t,label="no CV")
plt.plot(t_space,cv_1st,color='k',label="1st order polynomial")
plt.legend()
f3.savefig('CV_1st_order.pdf',bbox_inches='tight',dpi=100)
plt.show()

print('estimation of log phi1: (No CV)', np.sum(est_log_phi1)+log_phi0)
print('estimation of log phi1: (1st polynomial)', np.sum(cv_1st_grid)+log_phi0)
print('true value of log phi1:', true_log_phi1)

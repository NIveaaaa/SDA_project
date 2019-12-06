# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:39:22 2019

@author: z5187692
"""
import torch
import numpy as np
import math
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import kstest
import scipy.integrate as intergrate
import matplotlib.pyplot as plt
import statsmodels.api as sm
import SDAutility as Sutil
#%%
class Univariate_Normal_Path_sample:
    
    def __init__(self,smin,smax,t,mu,sigma,\
                 nsample = None, use_CV = False, CV_order = 1,use_qmc=False):
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
        if not use_qmc:
            self.z = truncnorm.rvs(a,b,loc=self.mu,\
                               scale = self.sigma_at_t,\
                               size=self.nsample)
        else:
            soboleng = torch.quasirandom.SobolEngine(dimension = 1,scramble=True)
            self.z = truncnorm.ppf(soboleng.draw(self.nsample),a,b,loc=self.mu,\
                                   scale=self.sigma_at_t)
        self.phi_theta = None
        self.h_theta = None
    
    def get_phi_theta(self):
        #self.phi_theta = -0.5*np.log(2*np.pi)-np.log(self.sigma)-\
        #0.5*np.power((self.z - self.mu)/self.sigma,2)
        self.phi_theta = norm.logpdf(self.z,loc=self.mu,scale=self.sigma)
    def get_expection(self):
        self.get_phi_theta()
        if self.use_CV:
            self.get_cv()
            self.expectation = np.mean(self.phi_theta-self.h_theta)
        else:
            self.expectation = np.mean(self.phi_theta)
        #self.expectation = np.mean(norm.logpdf(self.z,loc=self.mu,scale=self.sigma))
    
    def get_cv(self):
        self.deriv  = -1*(self.z - self.mu)/np.power(self.sigma_at_t,2)
                
        if (self.CV_order==1):

            x = Sutil.getX_order1(self.deriv)
            x_theta = sm.add_constant(x)

        # order 2
        if (self.CV_order==2):
            x = Sutil.getX_order2(self.z.reshape((self.nsample,1)),self.deriv.reshape((self.nsample,1)))
            x_theta = sm.add_constant(x)
            
            
        model = sm.OLS(self.phi_theta,x_theta)
        results = model.fit()
        #print(results.params)
        #print(np.multiply(x,results.params[1:]))
        fitted_value = np.multiply(x,results.params[1:]).reshape((self.nsample,self.CV_order))
        self.h_theta = np.sum(fitted_value,axis=1)
        #print(self.h_theta)
        
            # to control for multi-collinearity, add ep*I to X'X
        #m = np.linalg.inv(np.dot(np.transpose(x_theta),x_theta)) +\
        #np.diag(np.repeat(1e-6,self.CV_order+1))
        #params = np.dot(np.dot(m,np.transpose(x_theta)), self.phi_theta)
        #print(params)
        #self.h_theta = -np.sum(np.multiply(x,params[1:]),axis=1)
       
def intergrand(smin,smax,t,mu,sigma,nsample=100):
    uni = Univariate_Normal_Path_sample(smin,smax,t,mu,sigma)
    uni.get_expection()
    return(uni.expectation)


def log_phi1(smin,smax,mu,sigma):
    return(np.log(norm.cdf(smax,loc=mu,scale=sigma)\
                    -norm.cdf(smin,loc=mu,scale=sigma)))
    

def compute_integral_1st(smin,smax,mu,sigma,t_space,nsample=1000,use_CV=False,CV_order = 1,use_qmc=False):
    end_point = np.zeros(t_space.shape[0])
    for i in np.arange(start=0,stop=t_space.shape[0],step=1):
        uni = Univariate_Normal_Path_sample(smin,smax,t_space[i],mu,sigma,nsample,use_CV, CV_order,use_qmc)
        uni.get_expection() # compute the expecation
        end_point[i] = uni.expectation
    left_point = end_point[0:-1]
    right_point = end_point[1:]
    final_result = np.multiply(np.diff(t_space),np.mean([left_point,right_point],axis=0))
    return(end_point,final_result)


def compute_integral_2nd(smin,smax,mu,sigma,t_space,nsample=1000,use_CV=False,CV_order=1,use_qmc=False):
    mean_end_point = np.zeros(t_space.shape[0])
    var_end_point = np.zeros(t_space.shape[0])
    for i in np.arange(start=0,stop=t_space.shape[0],step=1):
        uni = Univariate_Normal_Path_sample(smin,smax,t_space[i],mu,sigma,nsample,use_CV,CV_order,use_qmc)
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
    a = (smin-mu)*np.sqrt(t/sigma)
    b = (smax - mu)*np.sqrt(t/sigma)
    """
    alpha = (a-0)/sigma0
    beta = (b-0)/sigma0
    phi_alpha = norm.pdf(alpha)
    phi_beta = norm.pdf(beta)
    Phi_alpha = norm.cdf(alpha)
    Phi_beta = norm.cdf(beta)
    Z = Phi_beta - Phi_alpha
    mean_value = 0 + (phi_alpha - phi_beta)/Z*sigma0
    var_value = np.power(sigma0,2)*(1+(alpha*phi_alpha-beta*phi_beta)/Z\
                         + np.power((phi_alpha-phi_beta)/Z,2))
    second_m = var_value+np.power(mean_value,2)
    """

    mean,var = truncnorm.stats(a,b,moments='mv',loc=0,scale=sigma0)
    second_m = np.power(mean,2)+var
    return(-second_m-0.5*np.log(2*np.pi)-np.log(sigma))
                                             
#%%    

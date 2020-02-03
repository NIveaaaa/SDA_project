# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:29:44 2020

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

#%%

"""
this function computes the path sampling estimator. A_m
input:
    smin: min value of observation
    smax: max value of observation
    nobs: number of observations
    temp: temperture t
    theta: parameter, theta[0]: mean, theta[1]: sd
    rand_nbrs: random numbers to be used in draws
    **kwargs: {approx_order: 1  or 2 } (to be completed)
    
output:
    A_m
    

procedure: 
    1. sampling z~ g*(z;\theta), where g* is a truncated normal with parameter:
        (theta[0],theta[1],smin,smax)
    2. for each temperature t: 
        compute the equation:
            E(C2/C1 * q_t(.) log g(.)/ g(.)) by plugging z and compute the average
            
    after obtained E(.) at temperature t, do numerical intergration, 
"""

class univariate_path_sampler2:
    def __init__(self,S,nobs,temp,theta,rand_nbrs,**kwargs):
        self.smin = S[0]
        self.smax = S[1]
        self.nobs = nobs
        self.mu = theta[0]
        self.sigma = theta[1]
        self.temp = temp
        self.theta = theta
        self.rand_nbrs = rand_nbrs
        self.approx_order = kwargs['approx_order']
        self.C1 =  math.lgamma(self.nobs+1)- math.lgamma(self.nobs-2+1)
        self.C2 = np.sum(norm.logpdf([self.smin,self.smax],loc=self.mu,scale=self.sigma))
        self.a = (self.smin - self.mu)/self.sigma
        self.b = (self.smax - self.mu)/self.sigma
        self.z = self.sample_z()
        self.g = norm.pdf(self.z,self.mu,self.sigma)/(norm.cdf(self.b)-norm.cdf(self.a))
        
        
    def sample_z(self):
        z = truncnorm.ppf(self.rand_nbrs,self.a,self.b,loc=self.mu,scale=self.sigma)
        return z
    
    def calc_log_g(self,t):
        #
        """
        term1 = np.sqrt(t)*(norm.cdf(self.b)-norm.cdf(self.a))/\
            (norm.cdf(self.b*np.sqrt(t) - norm.cdf(self.a*np.sqrt(t))))
        term2 = -0.5*np.log(2*np.pi)- np.log(self.sigma)-0.5*np.power(z-self.mu,2)/np.power(self.sigma,2)
        term3= np.exp(-0.5*np.power(z-self.mu,2)/np.power(self.sigma,2)*(t-1))
        log_g = term1*term2*term3
        
        
        """
        """
        the following approach is correct
        z = norm.rvs(size=1000)*self.sigma+self.mu
        qt = truncnorm.pdf(z,self.a*np.sqrt(t),self.b*np.sqrt(t),self.mu,self.sigma/np.sqrt(t))
        logg = norm.logpdf(z,self.mu,self.sigma)
        g = norm.pdf(z,self.mu,self.sigma)
        log_g  = qt/g*logg
        """
        
        """
        qt = truncnorm.pdf(z,self.a*np.sqrt(t),self.b*np.sqrt(t),self.mu,self.sigma/np.sqrt(t))
        logg = norm.logpdf(z,self.mu,self.sigma)
        g = truncnorm.pdf(z,self.a,self.b,self.mu,self.sigma)
        log_g  = qt/g*logg
        """
        
        """
        qt = norm.pdf(z,self.mu,self.sigma/np.sqrt(t))/\
            (norm.cdf(self.b*np.sqrt(t))-norm.cdf(self.a*np.sqrt(t)))
        logg = norm.logpdf(z,self.mu,self.sigma)
        g = truncnorm.pdf(z,self.a,self.b,self.mu,self.sigma)
        log_g  = qt/g*logg
        """
    
        qt = norm.pdf(self.z,self.mu,self.sigma/np.sqrt(t))/\
            (norm.cdf(self.b*np.sqrt(t))-norm.cdf(self.a*np.sqrt(t)))
        logg = norm.logpdf(self.z,self.mu,self.sigma)
        log_g  = qt/self.g*logg
        
        
        E_log_g = np.mean(log_g)
        var_log_g = np.std(log_g)
        return E_log_g, var_log_g


    def numerical_integral(self):
        temp_len = self.temp.shape[0]
        mean_end_point = np.zeros(temp_len)
        var_end_point = np.zeros(temp_len)
        
        if len(self.rand_nbrs)==0:
            return np.NaN
        else:
            for i in np.arange(start=0,stop=temp_len,step=1):
                mean_end_point[i],var_end_point[i] = self.calc_log_g(self.temp[i])
            diff_t_space = np.diff(self.temp)
            
            first_approx = np.multiply(diff_t_space,\
                                      np.mean([mean_end_point[0:-1],\
                                               mean_end_point[1:]],axis=0))
            
            quad_approx = first_approx - np.multiply(np.power(diff_t_space,2),\
                                 np.diff(var_end_point))/12
            
            if self.approx_order ==1:
                return np.sum(first_approx)
            if self.approx_order==2:
                return np.sum(quad_approx)
    
    def loglik_symbol(self):
        log_L = self.numerical_integral()+np.log(self.smax-self.smin)
        print('est:',log_L)
        return self.C1+self.C2+(self.nobs -2)*log_L
    
    def true_log_L(self):
        true_logL = np.log(norm.cdf(self.smax,loc=self.mu,scale=self.sigma)\
                    -norm.cdf(self.smin,loc=self.mu,scale=self.sigma))
        print('true logL', true_logL)
        return self.C1+self.C2+(self.nobs-2)*true_logL
        

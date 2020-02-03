# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:59:58 2020

@author: z5187692

This scripts use BlockPoisson estimator to do pseudo marginal MCMC  on a simple
model.

Model: 
    y = mu + error, error ~ N(0, sigma^2)
    
Prior: 
    mu ~ N(0, 10^2)
    sigma^2 proportional to 1/sigma^2

"""

import os
os.chdir('C:\\Users\z5187692\OneDrive - UNSW\SDA\SDA_project')
import numpy as np
import scipy as sp
import numpy.random as npr
import scipy.stats as sps
import sobol_seq
import univariate_path_sampler as ups
import copy
import BlockPoisson_SDA as BP
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
#%% set up parameters
npr.seed(100)
M_BP = 30 # Number of symbols (= number of blocks in the block-Poisson)
Nobs_BP = np.repeat(100,M_BP) # Number of observations per symbol
theta_true = np.array([0,1]) # mu and sigma
d = 1 # dimension of the integral we are estimating with the Sobol sequence.
kwargs = {'M_BP' : M_BP,  'rho_BP' : BP.rho_BP, 'd' : d, 'M' : BP.M, 'approx_order': 1}
temp = np.log(np.logspace(start=0.001,stop=1,num=200,base=np.e)) # temperature
#%% simulate observations

def extract_symbol(nsymbol, nobs, mu, sig):
    # sample is a list object
    sample = [norm.rvs(size = nobs[i])*sig + mu for i in range(nsymbol)]
    smin = [np.min(item) for item in sample]
    smax = [np.max(item) for item in sample]
    S = np.vstack((smin,smax)).T
    return S

# S is M_BP by 2 array with [smin,smax] for each symbol
global S 
S = extract_symbol(M_BP, Nobs_BP, theta_true[0],theta_true[1])

#%% prepare for MCMC, peudo-marginal MCMC
niter = 2000
theta_MCMC = np.zeros(shape=(niter,2))
theta_BPMCMC = np.zeros(shape=(niter,2))
theta_MCMC_trueL = np.zeros(shape=(niter,2))
#%% initialise value for mu, sigma
theta_MCMC[0] = theta_BPMCMC[0] = np.random.uniform(0,1,size=2)
# given theta, propose theta' (uniform distribution)
# in such case q(theta|theta') = q(theta'|theta)
def generate_new_proposal(theta,step=[0.2,0.2]):
    return np.random.uniform(size=2,low=-0.5,high=0.5) *step + theta
 


def A_m_hat(theta, S, Nobs_BP,Random_nbrs):
    """
    Todo: Implement A_m_hat. This is the path-sampling estimator. The code inputs Random_nbrs (which may be Quasi or Standard) uniforms, which are used to compute the path sampling
          estimator. The code also inputs S (the rectangle) and theta (the parameter) 
          input: S (boundary points), Nobs_BP (num of observations) theta(parameter), nobs (number of observations per symbol)
          output: A_m_hat 
    """
    up =ups.univariate_path_sampler(S, Nobs_BP, temp, theta, Random_nbrs, **kwargs)
    return up.numerical_integral()
#%% classic MCMC with estimated log-likelihood
 
# theta: mu, log (sigma)
def MCMC_hat_logL():
    for i in range(niter-1):
        
        mu_current = theta_MCMC[i,0]
        log_sigma_current = theta_MCMC[i,1]
    
        # change of varialbe l(s;mu,log(sigma)) = l(s;mu,sigma)+log(sigma)
        log_lik_current = np.sum([A_m_hat(np.array([mu_current,np.exp(log_sigma_current)]),\
                                          S[j], Nobs_BP[j], sps.uniform.rvs(size=BP.M)) \
                                  for j in range(len(Nobs_BP))])+log_sigma_current
        
        log_prior_current = sps.norm.logpdf(mu_current,loc=0,scale=10)
        theta_prop = generate_new_proposal([mu_current,log_sigma_current])
        mu_prop = theta_prop[0]
        log_sigma_prop = theta_prop[1]
        log_lik_prop = np.sum([A_m_hat(np.array([mu_prop,np.exp(log_sigma_prop)]), S[j], Nobs_BP[j], \
                                       sps.uniform.rvs(size=BP.M)) \
                               for j in range(len(Nobs_BP))])+log_sigma_prop
        log_prior_prop = sps.norm.logpdf(mu_prop,loc=0,scale=10)
        accept_ratio = np.min(log_lik_prop+ log_prior_prop-log_lik_current-log_prior_current,0)
        u = sps.uniform.rvs(size=1)
        accepted = np.log(u)<accept_ratio
        print('i:', i, " accepted:", accepted)
        if accepted: theta_current = theta_prop
        theta_MCMC[i+1] = theta_current
    
    #np.save('theta_MCMC.npy',theta_MCMC)
#%% classic MCMC with true log-likelihood plug in
# it works well
# theta: mu, log(sigma)
def MCMC_true_logL():
    for i in range(niter):
        theta_current = theta_MCMC_trueL[i]
        mu_current = theta_current[0]
        log_sigma_current= theta_current[1]
        sigma_current = np.exp(log_sigma_current)
        ll_curr = log_sigma_current
        for j in range(M_BP):
            C1 =  math.lgamma(Nobs_BP[j]+1)- math.lgamma(Nobs_BP[j]-1)
            C2 = np.sum(norm.logpdf(S[j],loc=mu_current,scale=sigma_current))
            true_logL = np.log(norm.cdf(S[j,1],loc=mu_current,scale=sigma_current)\
                        -norm.cdf(S[j,0],loc=mu_current,scale=sigma_current))
            ll_curr = ll_curr+ C1+C2+(Nobs_BP[j]-2)*true_logL
        
        log_prior_current = sps.norm.logpdf(mu_current,loc=0,scale=10)
        
        
        theta_prop = generate_new_proposal([mu_current, log_sigma_current])
        mu_prop = theta_prop[0]
        log_sigma_prop = theta_prop[1]
        sigma_prop = np.exp(log_sigma_prop)
        ll_prop = log_sigma_prop
        for j in range(M_BP):
            C1 =  math.lgamma(Nobs_BP[j]+1)- math.lgamma(Nobs_BP[j]-1)
            C2 = np.sum(sps.norm.logpdf(S[j],loc=mu_prop,scale=sigma_prop))
            true_logL = np.log(sps.norm.cdf(S[j,1],loc=mu_prop,scale=sigma_prop)\
                        -sps.norm.cdf(S[j,0],loc=mu_prop,scale=sigma_prop))
            ll_prop = ll_prop+ C1+C2+(Nobs_BP[j]-2)*true_logL
            
        log_prior_prop = sps.norm.logpdf(mu_prop,loc=0,scale=10)
        accept_ratio = np.min(ll_prop + log_prior_prop - ll_curr - log_prior_prop,0)
        u = sps.uniform.rvs(size=1)
        accepted = np.log(u)<accept_ratio
        print('i:', i, " accepted:", accepted)
        if accepted: theta_current = theta_prop
        theta_MCMC_trueL[i+1] = theta_current
        
#%% BP-MCMC

# fix an iteration i:
        
# 1. initialise u:
theta_cur = [0,0]
U,kappa = BP.init_random_numbers(BP.isQuasi, **kwargs)
U_prop = BP.PropU_given_currentU(U,kappa,BP.isQuasi,**kwargs)
logEst_cur,isNeg_cur,anyNeg_cur,lb_cur = BP.log_abs_BP_estimator([0,1],S,U,BP.a_BP,**kwargs)   

theta_prop = generate_new_proposal(theta_cur)
logEst_prop,isNeg_prop,anyNeg_prop,lb_prop = BP.log_abs_BP_estimator([theta_prop[0],np.exp(theta_prop[1])],\
                                                                     S,U_prop,BP.a_BP,**kwargs)

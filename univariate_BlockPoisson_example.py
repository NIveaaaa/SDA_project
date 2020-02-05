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


#%% initialise value for mu, sigma
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
    return up.loglik_symbol()
#%% classic independent pseudo marignal MCMC with estimated log-likelihood, it works, though not very good
# Note Plug in exp(A.hat) as an estimator of exp(A), which is BIASED.
# theta: mu, log (sigma)
def MCMC_hat_logL(niter):
    theta_MCMC = np.zeros(shape=(niter,2))
    theta_MCMC[0] = sps.uniform.rvs(size=2)
    for i in range(1,niter):
        theta_current = theta_MCMC[i]
        mu_current = theta_current[0]
        log_sigma_current = theta_current[1]
    
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
        theta_MCMC[i] = theta_current
    return theta_MCMC
    #np.save('theta_MCMC.npy',theta_MCMC)
theta_MCMC_hat = MCMC_hat_logL(50)
#%% classic PM - MCMC with true likelihood plug in
# it works well
# theta: mu, log(sigma)
def MCMC_true_logL(niter):
    theta_MCMC_trueL = np.zeros(shape=(niter,2))
    theta_MCMC_trueL[0] = sps.uniform.rvs(size=2)
    for i in range(niter-1):
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
    return theta_MCMC_trueL
test_trueL = MCMC_true_logL(1000)

#%% classic PM with Blocking 
    # note plug in with exp(A.hat), which is NOT an unbiased estimator of exp(A)
def PM_MCMC_blocking(niter,rho_BP = 0.8):
    theta_PM_MCMC_B = np.zeros(shape=(niter,2))
    theta_PM_MCMC_B[0] = sps.uniform.rvs(size=2)
    kappa = np.ceil((1 - rho_BP)*M_BP).astype('int')
    U = [sps.uniform.rvs(size = (BP.M, d)) for item in range(M_BP)]
    for i in range(niter-1):
        
        theta_current = theta_PM_MCMC_B[i]
        mu_current = theta_current[0]
        log_sigma_current = theta_current[1]
    
        # change of varialbe l(s;mu,log(sigma)) = l(s;mu,sigma)+log(sigma)
        log_lik_current = [A_m_hat(np.array([mu_current,np.exp(log_sigma_current)]),\
                                          S[j], Nobs_BP[j], U[j]) \
                                  for j in range(len(Nobs_BP))]
        
        log_prior_current = sps.norm.logpdf(mu_current,loc=0,scale=10)
        theta_prop = generate_new_proposal([mu_current,log_sigma_current])
        mu_prop = theta_prop[0]
        log_sigma_prop = theta_prop[1]
        
        U_prop = copy.deepcopy(U)
        BlocksToChange = npr.choice(M_BP, kappa, replace=False)
        
        log_lik_prop = log_lik_current
        for BlockToChange in BlocksToChange:
            U_prop[BlockToChange] = sps.uniform.rvs(size=(BP.M,d))
        
        log_lik_prop = [A_m_hat(np.array([mu_prop,np.exp(log_sigma_prop)]),\
                                          S[j], Nobs_BP[j], U_prop[j]) for j in range(len(Nobs_BP))]
                

        log_prior_prop = sps.norm.logpdf(mu_prop,loc=0,scale=10)
        accept_ratio = np.min(np.sum(log_lik_prop)+ log_prior_prop  + log_sigma_prop\
                              - np.sum(log_lik_current)-log_prior_current - log_sigma_current,0)
        u = sps.uniform.rvs(size=1)
        accepted = np.log(u)<accept_ratio
        print('i:', i, " accepted:", accepted)
        if accepted: 
            theta_current = theta_prop
            U = U_prop
        theta_PM_MCMC_B[i] = theta_current
    return theta_PM_MCMC_B

test_PM_MCMC_B = PM_MCMC_blocking(200)

#%% Poisson PM-MCMC without blocking
    
niter = 200
theta_PMCMC = np.zeros(shape=(niter,2))
theta_PMCMC[0] = sps.uniform.rvs(size=(2))
sign_L_PMCMC = np.ones(niter)
#  initialise u:
U,kappa = BP.init_random_numbers(BP.isQuasi, **kwargs)
            
for i in range(1,niter):
    if i%100 == 0:
        print('iterations:',i)
    U_temp = U
    
    
    # current theta 
    theta_cur = theta_PMCMC[i-1]
    logEst_cur,isNeg_cur,_,_ = BP.log_abs_BP_estimator([theta_cur[0],np.exp(theta_cur[1])],S,U,BP.a_BP,**kwargs)   
    prior_cur = sps.norm.logpdf(theta_cur[0],0,10)
    
    # generate theta proposal
    U_prop,_ = BP.init_random_numbers(BP.isQuasi, **kwargs)
    theta_prop = generate_new_proposal(theta_cur)
    logEst_prop,isNeg_prop,_,_ = BP.log_abs_BP_estimator([theta_prop[0],np.exp(theta_prop[1])],S,U_prop,BP.a_BP,**kwargs)
    prior_prop = sps.norm.logpdf(theta_prop[0],0,10)
    

    dif_logl = logEst_prop + theta_prop[1]+prior_prop - logEst_cur - theta_cur[1]-prior_cur
    
    ar = np.min(dif_logl, 0)
    u = sps.uniform.rvs(size=1)
    accepted = np.log(u)<ar
    
    if accepted:
        theta_cur = theta_prop
        sign_L_PMCMC[i] = isNeg_prop
    else:
        sign_L_PMCMC[i] = isNeg_cur
        U = U_temp
    theta_PMCMC[i] = theta_cur

#np.save('output/theta_BPMCMC.npy',theta_BPMCMC)
#np.save('output/sign_L.npy',sign_L)

sign_L_PMCMC[sign_L_PMCMC==1]  = -1
sign_L_PMCMC[sign_L_PMCMC==0] = 1

np.sum(theta_PMCMC[:,0]*sign_L_PMCMC)/np.sum(sign_L_PMCMC)
np.sum(theta_PMCMC[:,1]*sign_L_PMCMC)/np.sum(sign_L_PMCMC)


#%% BP-MCMC
    
niter = 2000
theta_BPMCMC = np.zeros(shape=(niter,2))
theta_BPMCMC[0] = sps.uniform.rvs(size=(2))
sign_L = np.ones(niter)
#  initialise u:
U,kappa = BP.init_random_numbers(BP.isQuasi, **kwargs)
A_m_hats = copy.deepcopy(U)

for r in range(len(U)):
    count = len(U[r])
    if count == 0:
        continue
    else:
        for j in range(count):
            A_m_hats[r][j] = A_m_hat(theta_BPMCMC[0],S[r],Nobs_BP[r],np.asarray(U[r][j]).flatten())
            
for i in range(1,niter):
    if i%100 == 0:
        print('iterations:',i)
    U_temp = U
    
    theta_cur = theta_BPMCMC[i-1]
    # sample u_proposal given U
    U_prop,BlocksToChange= BP.PropU_given_currentU(U,kappa,BP.isQuasi,**kwargs)
    logEst_cur,isNeg_cur,A_m_hats_cur,anyNeg_cur,lb_cur = BP.log_abs_BP_estimator_new([theta_cur[0],np.exp(theta_cur[1])],S,U,np.array([]),A_m_hats,BP.a_BP,**kwargs)   
    prior_cur = sps.norm.logpdf(theta_cur[0],0,10)
    
    # generate theta proposal
    theta_prop = generate_new_proposal(theta_cur)
    logEst_prop,isNeg_prop,A_m_hats_prop,anyNeg_prop,lb_prop = BP.log_abs_BP_estimator_new([theta_prop[0],np.exp(theta_prop[1])],S,U_prop,BlocksToChange,A_m_hats,BP.a_BP,**kwargs)
    prior_prop = sps.norm.logpdf(theta_prop[0],0,10)
    dif_logl = logEst_prop + np.exp(theta_prop[1])+prior_prop - logEst_cur - np.exp(theta_cur[1])-prior_cur
    
    ar = np.min(dif_logl, 0)
    u = sps.uniform.rvs(size=1)
    accepted = np.log(u)<ar
    
    if accepted:
        theta_cur = theta_prop
        sign_L[i] = isNeg_prop
        A_m_hats = A_m_hats_prop
    else:
        sign_L[i] = isNeg_cur
        U = U_temp
    theta_BPMCMC[i] = theta_cur

#np.save('output/theta_BPMCMC.npy',theta_BPMCMC)
#np.save('output/sign_L.npy',sign_L)

sign_L[sign_L==1]  = -1
sign_L[sign_L==0] = 1

np.sum(theta_BPMCMC[:,0]*sign_L)/np.sum(sign_L)
np.sum(theta_BPMCMC[:,1]*sign_L)/np.sum(sign_L)

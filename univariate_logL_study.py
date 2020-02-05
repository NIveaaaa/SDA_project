# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:50:46 2020

@author: z5187692
"""

import os
os.chdir('C:\\Users\z5187692\OneDrive - UNSW\SDA\SDA_project')
import numpy as np
import scipy as sp
import numpy.random as npr
import scipy.stats as sps
import sobol_seq
import univariate_path_sampler as ups
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import BlockPoisson_SDA as BP
#%%
def extract_symbol(nsymbol, nobs, mu, sig):
    # sample is a list object
    sample = [norm.rvs(size = nobs)*sig + mu for i in range(nsymbol)]
    smin = [np.min(item) for item in sample]
    smax = [np.max(item) for item in sample]
    S = np.vstack((smin,smax)).T
    return S

#%%
M = 30
Nobs = 100
theta_true=np.array([1,1])
kwargs = {'approx_order': 1}
temp = np.log(np.logspace(start=0.001,stop=1,num=200,base=np.e)) # temperature

S = extract_symbol(M,Nobs,theta_true[0],theta_true[1])
NDraws = 200
niter = 200
kwargs = {'approx_order':1}
#%%
true_l = 0

for j in range(M):
    C1 =  math.lgamma(Nobs+1)- math.lgamma(Nobs-1)
    smax = S[j][1]
    smin = S[j][0]
    C2 = np.sum(norm.logpdf([smin,smax],loc=theta_true[0],scale=theta_true[1]))
    true_logL = np.log(sps.norm.cdf(smax,loc=theta_true[0],scale=theta_true[1])\
                    -sps.norm.cdf(smin,loc=theta_true[0],scale=theta_true[1]))
    true_l = true_l + C1+C2+ (Nobs-2)*true_logL

print('true log likihood: ',true_l)

#%% 
hat_l = []
for i in range(niter):
    if i%10==0: print('iter',i)
    templ = 0
    for j in range(M):
        up = ups.univariate_path_sampler(S[j],Nobs,temp,theta_true,\
                                         sps.uniform.rvs(size=(NDraws,1)),**kwargs)
        templ = templ + up.loglik_symbol()
    hat_l.append(templ)
    
np.save('output/hat_l_M_30_Nobs_30_Ndraw_200_niter_200.npy',hat_l)

# the var(log L (..)) is 29, with mean -21..

#%%

hat_l_quasi = []

for i in range(niter):
    templ = 0
    for j in range(M):
        up = ups.univariate_path_sampler(S[j],Nobs,temp,theta_true,\
                                         BP.generate_Sobol(1,NDraws),**kwargs)
        templ = templ + up.loglik_symbol()
    hat_l_quasi.append(templ)

# the var log L(...) is 2.4 with mean -22

np.save('output/hat_l_M_30_Nobs_30_Ndraw_200_niter_200_Quasi.npy',hat_l_quasi)
    
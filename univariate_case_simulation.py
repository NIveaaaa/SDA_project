# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:17:31 2019

@author: z5187692

this file is the case study for univariate normal

"""

import numpy as np
import math
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import kstest
from scipy.special import factorial
import scipy.integrate as intergrate
import matplotlib.pyplot as plt
import statsmodels.api as sm
import univariate_case as uc
import random
from tempfile import TemporaryFile
#%%

"""
Road map
1. extract symbol for each group (by simulation)
2. compute the log likelihood for each symbol
3. compare the est log likelihood with true log likelihood and its variance
"""

def extract_symbol(nsymbol, nobs, mu, sig):
    # sample is a list object
    sample = [norm.rvs(size = nobs)*sig + mu for i in range(nsymbol)]
    smin = [np.min(item) for item in sample]
    smax = [np.max(item) for item in sample]
    return sample,smin,smax



def loglik_symbol(smin,smax,nobs,mu,sig,t_space,use_CV,CV_order):
    C1 = math.lgamma(nobs+1)- math.lgamma(nobs-2+1)
    C2 = np.sum(norm.logpdf([smin,smax],loc=mu,scale=sig))
    #
    _,logL = uc.compute_integral_1st(smin,smax,mu,sig,t_space,1000,use_CV,CV_order)
    C3 = np.sum(logL)+ np.log(smax-smin)
    logLik = C1 + C2 + (nobs -2)*C3
    return logLik,C3


def true_loglik_symbol(smin,smax,nobs,mu,sig):
    C1 = math.lgamma(nobs+1)- math.lgamma(nobs-2+1)
    C2 = np.sum(norm.logpdf([smin,smax],loc=mu,scale=sig))
    true_logL = uc.log_phi1(smin,smax,mu,sig)
    true_logLik = C1+C2+(nobs-2)*true_logL
    return true_logLik,true_logL

def one_iter(nobs,nsymbols=30,mu=1,sig=2,use_CV=False,CV_order=1):
    t_space = np.log(np.logspace(start=0.0001,stop=1,base=np.e,num=2000))
    random.seed(123)
    out,sminn,smaxx = extract_symbol(nsymbols,nobs,mu,sig)
    logLik_ind = np.zeros(nsymbols)
    C3_est = np.zeros(nsymbols)
    true_logLik_ind = np.zeros(nsymbols)
    C3_true = np.zeros(nsymbols)
    for i in range(nsymbols):
        logLik_ind[i],C3_est[i] = loglik_symbol(sminn[i],smaxx[i],nobs,1,2,t_space,use_CV,CV_order)
        true_logLik_ind[i],C3_true[i] = true_loglik_symbol(sminn[i],smaxx[i],nobs,1,2)
    
    #print('true log lik:',np.sum(true_logLik_ind))
    #print('est log lik:',np.sum(logLik_ind))
    return np.sum(true_logLik_ind),np.sum(logLik_ind)

nobs = [10,50,100,500,1000,1500,2000,5000,10000]
#nobs = [10,50,100]
iters = 10
res_est = np.zeros((len(nobs),iters))
res_true = np.zeros((len(nobs),iters))

"""
for j in range(len(nobs)):
    for k in range(iters):
        res_true[j,k], res_est[j,k]=one_iter(nobs[j],nsymbols=30,mu=1,sig=2,\
        use_CV = False, CV_order=1)

"""
res_est = np.load('result_est_10iter.npy')
res_true = np.load('result_true_10iter.npy')

diff_res = abs(np.divide(res_est - res_true,res_true)*100)

diff_mean = np.mean(diff_res,axis=1)
diff_sd = np.std(diff_res,axis = 1)

f0 = plt.figure(figsize=(6,6))
plt.plot(range(len(nobs)),diff_mean,'-o',label='mean')
plt.xticks(range(len(nobs)),nobs)
plt.xlabel('observations within each symbol')
plt.ylabel('|(est - true)/true| (%)')
plt.fill_between(range(len(nobs)), diff_mean - diff_sd,\
                      diff_sd + diff_mean, alpha = 0.1, color="r",
                      label='mean+/- sd')
plt.legend()
f0.savefig('result_diff_nobs.pdf')


res_est_cv_1 = np.zeros((len(nobs),iters))
res_true_cv_1 = np.zeros((len(nobs),iters))

for j in range(len(nobs)):
    for k in range(iters):
        res_true_cv_1[j,k], res_est_cv_1[j,k]=one_iter(nobs[j],nsymbols=30,mu=1,sig=2,\
                use_CV = True, CV_order=1)

diff_res_cv = abs(np.divide(res_est_cv_1-res_true_cv_1,res_true_cv_1)*100)
diff_mean_cv = np.mean(diff_res_cv,axis=1)
#if __name__ == "__main__": 			  		 			     			  	   		   	  			  	
    
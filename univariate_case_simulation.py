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



def loglik_symbol(smin,smax,nobs,mu,sig,t_space,use_CV,CV_order,use_qmc=False):
    C1 = math.lgamma(nobs+1)- math.lgamma(nobs-2+1)
    C2 = np.sum(norm.logpdf([smin,smax],loc=mu,scale=sig))
    #
    _,logL = uc.compute_integral_1st(smin,smax,mu,sig,t_space,1000,use_CV,CV_order,use_qmc)
    C3 = np.sum(logL)+ np.log(smax-smin)
    logLik = C1 + C2 + (nobs -2)*C3
    return logLik,C3


def true_loglik_symbol(smin,smax,nobs,mu,sig):
    C1 = math.lgamma(nobs+1)- math.lgamma(nobs-2+1)
    C2 = np.sum(norm.logpdf([smin,smax],loc=mu,scale=sig))
    true_logL = uc.log_phi1(smin,smax,mu,sig)
    true_logLik = C1+C2+(nobs-2)*true_logL
    return true_logLik,true_logL

def one_iter(nobs,nsymbols=1,mu=1,sig=2,use_CV=False,CV_order=1,use_qmc=False):
    t_space = np.log(np.logspace(start=0.0001,stop=1,base=np.e,num=2000))
    random.seed(123)
    out,sminn,smaxx = extract_symbol(nsymbols,nobs,mu,sig)
    logLik_ind = np.zeros(nsymbols)
    C3_est = np.zeros(nsymbols)
    true_logLik_ind = np.zeros(nsymbols)
    C3_true = np.zeros(nsymbols)
    for i in range(nsymbols):
        logLik_ind[i],C3_est[i] = loglik_symbol(sminn[i],smaxx[i],nobs,1,2,t_space,use_CV,CV_order,use_qmc)
        true_logLik_ind[i],C3_true[i] = true_loglik_symbol(sminn[i],smaxx[i],nobs,1,2)
    
    #print('true log lik:',np.sum(true_logLik_ind))
    #print('est log lik:',np.sum(logLik_ind))
    return np.sum(true_logLik_ind),np.sum(logLik_ind)
def testcase01():
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

# this case split sample into K folds
def testcase02(K=5,mu=0,sigma=1,nobs=100):
    t_space = np.log(np.logspace(start=0.0001,stop=1,base=np.e,num=2000))
    if (nobs%K!=0):
        print('num of obs can not be split into equal sized subsample')
        #return
    sample = norm.rvs(mu, sigma, size=nobs)
    np.random.shuffle(sample)
    sample_split = np.split(sample,K)
    smin_split = [np.min(item) for item in sample_split] 
    smax_split = [np.max(item) for item in sample_split]
    smin = np.min(smin_split)
    smax = np.max(smax_split)
    logLik_each = np.zeros(K)
    true_logLik_each = np.zeros(K)
    for k in range(K):
        logLik_each[k],_ = loglik_symbol(smin_split[k], smax_split[k], nobs/K, mu, sigma, t_space, False, 0, True)
        true_logLik_each[k],_ = true_loglik_symbol(smin_split[k],smax_split[k],nobs/K,mu,sigma)
    logLik_split = np.sum(logLik_each)
    logLik_whole,_ = loglik_symbol(smin,smax,nobs,mu,sigma,t_space,False,0,True)
    
    true_logLik,_ =  true_loglik_symbol(smin,smax,nobs,mu,sigma)
    print(logLik_split, logLik_whole,true_logLik)
    # true log likelihood does not equal to sum of symbol...

# this is the replicate of case01, the difference is to use qmc instead
def testcase03():
    nobs = [10,50,100,500,1000,1500,2000,5000,10000]
    #nobs = [10,50,100]
    iters = 10
    res_est = np.zeros((len(nobs),iters))
    res_true = np.zeros((len(nobs),iters))
    
    """
    for j in range(len(nobs)):
        for k in range(iters):
            res_true[j,k], res_est[j,k]=one_iter(nobs[j],nsymbols=30,mu=1,sig=2,\
            use_CV = False, CV_order=1,use_qmc=True)
    """
    #np.save('result_est_10_iter_qmc.npy',res_est)
    #np.save('result_true_10_iter_qmc.npy',res_true)
    
    # following code are for plots
    res_est = np.load('result_est_10_iter_qmc.npy')
    res_true = np.load('result_true_10_iter_qmc.npy')
    
    res_est_mc = np.load('result_est_10iter.npy')
    res_true_mc = np.load('result_true_10iter.npy')
    
    ratio_qmc = abs(np.divide(res_est,res_true))
    ratio_mc = abs(np.divide(res_est_mc,res_true_mc))
    
    lb_mc = np.mean(ratio_mc,axis=1) - 2*np.std(ratio_mc,axis=1)
    ub_mc = np.mean(ratio_mc,axis=1) + 2*np.std(ratio_mc,axis=1)
    
    
    lb_qmc = np.mean(ratio_qmc,axis=1) - 2*np.std(ratio_qmc,axis=1)
    ub_qmc = np.mean(ratio_qmc,axis=1) + 2*np.std(ratio_qmc,axis=1)

    f0 = plt.figure(figsize=(6,6))
    plt.plot(range(len(nobs)),np.mean(ratio_qmc,axis =1),'-o',label='qmc mean')
    plt.plot(range(len(nobs)),np.mean(ratio_mc,axis =1 ),'-o',label='mc_mean')
    plt.xticks(range(len(nobs)),nobs)
    plt.xlabel('observations within each symbol')
    plt.ylabel('|ratio of est/true |')
    plt.fill_between(range(len(nobs)), lb_mc,\
                          ub_mc, alpha = 0.1, color="r",
                          label='mean+/- 2*sd, MC')
    plt.fill_between(range(len(nobs)), lb_qmc,\
                          ub_qmc, alpha = 0.1, color="b",
                          label='mean+/- 2*sd, QMC')
    plt.legend()
    f0.savefig('result_diff_nobs_mc_qmc.pdf')
    
# this test case try different symbols
def testcase04():
    nsymb = [5,10,30,50,100,200,500]
    iters = 10
    res_est = np.zeros((len(nsymb),iters))
    res_true = np.zeros((len(nsymb),iters))
    

    for j in range(len(nsymb)):
        for k in range(iters):
            res_true[j,k], res_est[j,k]=one_iter(500,nsymbols=nsymb[j],mu=1,sig=2,\
            use_CV = False, CV_order=0,use_qmc=True)
    
    # bias 
    ratio_ = res_est/res_true
    ratio_mean = np.mean(ratio_,axis=1)
    ratio_std = np.std(ratio_,axis=1)
    
    f0 = plt.figure(figsize=(6,6))
    plt.plot(range(len(nsymb)),ratio_mean,'-o', label="10 iterations")
    plt.fill_between(range(len(nsymb)),ratio_mean - ratio_std,\
                     ratio_mean + ratio_std,alpha = 0.1, color = "r",\
                         label="mean +/- std")
    plt.xticks(range(len(nsymb)),nsymb)
    plt.xlabel("number of symbols (500 observations within each symbol)")
    plt.ylabel("ratio between estimation over true value")
    f0.savefig('result_diff_symbs_qmc.pdf')
if __name__=="__main__":
    testcase03()
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:27:53 2019

@author: z5187692

this file serves for computing the unbiased estimate for log L by path sampling

univariate normal case

"""
import numpy as np
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import kstest
import scipy.integrate as intergrate
import matplotlib.pyplot as plt
import statsmodels.api as sm
import univariate_case as uc
        
smin = -2
smax = 2
mu = 0
sigma = 1


log_phi0 = np.log(smax-smin)


# est_log_phi1 = log_phi0 + result[0]
true_log_phi1 = uc.log_phi1(smin=smin,smax=smax,mu=mu,sigma=sigma)

t_space = np.log(np.logspace(start=0.0001,stop=1,base=np.e,num=2000))
#t_space = np.linspace(start=0.001,stop=1,num=2000)
est_log_phi1_per_t,est_log_phi1 = uc.compute_integral_1st(smin,smax,mu,sigma,t_space)

true_2nd_moment = [uc.true_second_moment(smin,smax,mu,sigma,t) for t in t_space]


print('estimation of log phi1:', np.sum(est_log_phi1)+log_phi0)
print('true value of log phi1:', true_log_phi1)

f = plt.figure(figsize=(6,6))
plt.plot(t_space,est_log_phi1_per_t,label='estimation')
plt.plot(t_space,true_2nd_moment,label='true value')
plt.xlabel('t')
plt.ylabel('expectation')
plt.legend()
#f.savefig('true_vs_estimation.pdf',bbox_iches = 'tight')
#plt.close()
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

print('the ratio between mean and std over 10000 MC estimates:',\
      abs(np.mean(est_log_phi1_1st_app)/np.std(est_log_phi1_1st_app)))

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
cv_1st,cv_1st_grid = uc.compute_integral_1st(smin,smax,mu,sigma,t_space,1000,True,1)

# 2nd order approximation does not work
cv_2nd,_, = uc.compute_integral_1st(smin,smax,mu,sigma,t_space,1000,True,2)

f3 = plt.figure(figsize=(6,6))
plt.plot(t_space,est_log_phi1_per_t,label="no CV")
plt.plot(t_space,cv_1st,color='k',label="1st order polynomial")
plt.plot(t_space,cv_2nd,color='y',label="2nd order polynomial")
plt.legend()
f3.savefig('CV_1st_order.pdf',bbox_inches='tight',dpi=100)
plt.show()

print('estimation of log phi1: (No CV)', np.sum(est_log_phi1)+log_phi0)
print('estimation of log phi1: (1st polynomial)', np.sum(cv_1st_grid)+log_phi0)
print('true value of log phi1:', true_log_phi1)



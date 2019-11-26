# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:27:53 2019

@author: z5187692

this file serves for computing the unbiased estimate for log L by path sampling

bivariate normal case

"""
import numpy as np
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import kstest
import scipy.integrate as intergrate
import matplotlib.pyplot as plt
import statsmodels.api as sm
import bivariate_case as bc
import SDAutility as Sutil
#%%
mu = np.array([0.,0.])
sigma = np.array([[1.,0.9],[0.9,1.]])

obs = np.random.multivariate_normal(mu,sigma,size=100)
plt.scatter(obs[:,0],obs[:,1])
smin = np.min(obs,axis=0)
smax = np.max(obs,axis=0)
#smin = np.array([-2,-2])
#smax = np.array([2,2])

log_phi0 = np.sum(np.log(smax-smin))

true_log_phi1 = bc.log_phi1(smin=smin,smax=smax,mu=mu,sigma=sigma)

num_t = 50
t_space = np.log(np.logspace(start=0.0001,stop=1,base=np.e,num=num_t))
#t_space = np.linspace(start=0.001,stop=1,num=num_t)

expect_order0 = np.zeros(num_t)
expect_order1 = np.zeros(num_t)
expect_order2 = np.zeros(num_t)

for t in range(t_space.size):
    bc0 = bc.Bivariate_Normal_Path_sample(smin,smax,t_space[t],mu,sigma,1000,False,0)
    bc1 = bc.Bivariate_Normal_Path_sample(smin,smax,t_space[t],mu,sigma,1000,True,1)
    bc2 = bc.Bivariate_Normal_Path_sample(smin,smax,t_space[t],mu,sigma,1000,True,2)
    bc0.get_expectation()
    bc1.get_expectation()
    bc2.get_expectation()
    expect_order0[t] = bc0.expectation
    expect_order1[t] = bc1.expectation
    expect_order2[t] = bc2.expectation

t0,final_t0 = Sutil.compute_integral_1st(expect_order0,t_space)
t1,final_t1 = Sutil.compute_integral_1st(expect_order1,t_space)
t2,final_t2 = Sutil.compute_integral_1st(expect_order2,t_space)

result_order1 = [final_t0,final_t1,final_t2] + log_phi0



print('true value of log phi1:', true_log_phi1)

print('estimation of approximation of order 1\n')
print('estimation  -poly order 0:', result_order1[0])
print('estimation  -poly order 1:', result_order1[1])
print('estimation  -poly order 2:', result_order1[2])


"""
t0_order1,t0_order2 = bc.compute_integral_2nd(smin,smax,mu,sigma,t_space,1000,False,0)
t1_order1,t1_order2 = bc.compute_integral_2nd(smin,smax,mu,sigma,t_space,1000,True,1)

result_order2 = [t0_order2,t1_order2] + log_phi0

print('estimation of approximaiton of order 2\n')
print('estimation  -poly order 0:', result_order2[0])
print('estimation  -poly order 1:', result_order2[1])
"""

#%% replication of 1000 times

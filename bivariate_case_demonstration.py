# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:27:53 2019

@author: z5187692

this file serves for computing the unbiased estimate for log L by path sampling

bivariate normal case
"""
import os
import numpy as np
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import kstest
import scipy.integrate as intergrate
import matplotlib.pyplot as plt
import statsmodels.api as sm
os.chdir('C:\\Users\\z5187692\\OneDrive - UNSW\\SDA\\SDA_project\\')
import bivariate_case as bc
import SDAutility as Sutil
import seaborn as sns


#%%

def bc_demonstration(mu,sigma,smin,smax,t_space):
    log_phi0 = np.sum(np.log(smax-smin))
    expect_order0 = np.zeros(t_space.size)
    #expect_order1 = np.zeros(num_t)
    #expect_order2 = np.zeros(num_t)
    
    for t in range(t_space.size):
        bc0 = bc.Bivariate_Normal_Path_sample(smin,smax,t_space[t],mu,sigma,1000,False,0)
        #bc1 = bc.Bivariate_Normal_Path_sample(smin,smax,t_space[t],mu,sigma,1000,True,1)
        #bc2 = bc.Bivariate_Normal_Path_sample(smin,smax,t_space[t],mu,sigma,1000,True,2)
        bc0.get_expectation()
        #bc1.get_expectation()
        #bc2.get_expectation()
        expect_order0[t] = bc0.expectation
        #expect_order1[t] = bc1.expectation
        #expect_order2[t] = bc2.expectation
    
    t0,final_t0 = Sutil.compute_integral_1st(expect_order0,t_space)
    #t1,final_t1 = Sutil.compute_integral_1st(expect_order1,t_space)
    #t2,final_t2 = Sutil.compute_integral_1st(expect_order2,t_space)
    
    #result_order1 = [final_t0,final_t1,final_t2] + log_phi0
    result_order1 = final_t0 + log_phi0
    
    #print('estimation of approximation of order 1\n')
    print('estimation  -poly order 0:', result_order1)
    #print('estimation  -poly order 1:', result_order1[1])
    #print('estimation  -poly order 2:', result_order1[2])
    return result_order1


"""
t0_order1,t0_order2 = bc.compute_integral_2nd(smin,smax,mu,sigma,t_space,1000,False,0)
t1_order1,t1_order2 = bc.compute_integral_2nd(smin,smax,mu,sigma,t_space,1000,True,1)

result_order2 = [t0_order2,t1_order2] + log_phi0

print('estimation of approximaiton of order 2\n')
print('estimation  -poly order 0:', result_order2[0])
print('estimation  -poly order 1:', result_order2[1])
"""



def test_01():
    mu = np.array([0.,0.])
    sigma = np.array([[1.,0.9],[0.9,1.]])
    obs = np.random.multivariate_normal(mu,sigma,size=100)
    smin = np.min(obs,axis=0)
    smax = np.max(obs,axis=0)
    num_t = 50
    t_space = np.log(np.logspace(start=0.0001,stop=1,base=np.e,num=num_t))
    #est = np.zeros(5)
    est = [bc_demonstration(mu,sigma,smin,smax,t_space) for i in range(2000)]
    tv = bc.log_phi1(smin,smax,mu,sigma)
    plot_to_hist(est,tv,mu,sigma,'test_01')
    est.append(tv)
    est.append(mu)
    est.append(sigma)
    est.append(smin)
    est.append(smax)
    with open('test_01_result.txt', 'w') as f:
        for item in est:
            f.write("%s\n" % item)
    


def test_02():
    mu = np.array([0.,0.])
    sigma = np.array([[1.,0.5],[0.5,1.]])
    obs = np.random.multivariate_normal(mu,sigma,size=100)
    smin = np.min(obs,axis=0)
    smax = np.max(obs,axis=0)
    num_t = 50
    t_space = np.log(np.logspace(start=0.0001,stop=1,base=np.e,num=num_t))
    #est = np.zeros(2000)
    est = [bc_demonstration(mu,sigma,smin,smax,t_space) for i in range(2000)]
    tv = bc.log_phi1(smin,smax,mu,sigma)
    plot_to_hist(est,tv,mu,sigma,'test_02')
    est.append(tv)
    est.append(mu)
    est.append(sigma)
    est.append(smin)
    est.append(smax)
    with open('test_02_result.txt', 'w') as f:
        for item in est:
            f.write("%s\n" % item)
            
            
def test_03():
    mu = np.array([0.,0.])
    sigma = np.array([[1.,0.1],[0.1,1.]])
    obs = np.random.multivariate_normal(mu,sigma,size=100)
    smin = np.min(obs,axis=0)
    smax = np.max(obs,axis=0)
    num_t = 50
    t_space = np.log(np.logspace(start=0.0001,stop=1,base=np.e,num=num_t))
    est = np.zeros(2000)
    est = [bc_demonstration(mu,sigma,smin,smax,t_space) for i in range(2000)]
    tv = bc.log_phi1(smin,smax,mu,sigma)
    plot_to_hist(est,tv,mu,sigma,'test_03')
    est.append(tv)
    est.append(mu)
    est.append(sigma)
    est.append(smin)
    est.append(smax)
    with open('test_03_result.txt', 'w') as f:
        for item in est:
            f.write("%s\n" % item)
            


        
def plot_to_hist(est,tv,mu,sigma,fn):
    f1 = plt.figure(figsize=(6,6))
    plt.hist(est,density=True)
    plt.axvline(tv, color='k', linestyle='dashed', linewidth=1,\
                label='true value')
    plt.axvline(np.mean(est),linestyle='dashed',\
                color='r',label='mean value')
    plt.legend()
    Suptitle = "mean value: "+ str(mu[0])+","+str(mu[1]) +","+"var1,var2,cov: "+\
    str(sigma[0,0])+","+str(sigma[1,1])+"," + str(sigma[0,1])
    Title = "10,000 replications [bivariate normal distribution]"
    print(Suptitle)
    plt.suptitle(Suptitle) 
    plt.title(Title)
    est_mean = "{0:.4f}".format(np.mean(est))
    est_std = "{0:.4f}".format(np.std(est))
    Mytext = "true mean:"+ "{0:.4f}".format(tv)+'\n'+ \
             "mean:"+ est_mean+'\n'+"std:"+est_std + '\n' +\
             "ratio mean/std:"+ "{0:.4f}".format(np.mean(est)/np.std(est))
    plt.text(np.min(est),30, Mytext )
    f1.savefig(fn+'.pdf',bbox_inches='tight')

def read_output(fn):
    f = open(fn,"r")
    if f.mode =='r':
        out = f.read()
    return out
    

#%% replication of 1000 times
if __name__ == "__main__":
    #test_01() #loading output instead
    #test_02() #loading output instead
    #test_03() #loading output inteaad
    test01 = read_output("test_01_result_R.txt")
    test01_res = np.fromstring(test01,sep='\n')
    test01_tv = test01_res[-1]
    test01_est = test01_res[0:-1]
    
    test02 = read_output("test_02_result_R.txt")
    test02_res = np.fromstring(test02,sep='\n')
    test02_tv = test02_res[-1]
    test02_est = test02_res[0:-1]
    
    test03 = read_output("test_03_result_R.txt")
    test03_res = np.fromstring(test03,sep='\n')
    test03_tv = test03_res[-1]
    test03_est = test03_res[0:-1]
    
    
    bias01 = test01_est-test01_tv
    bias02 = test02_est-test02_tv
    bias03 = test03_est-test03_tv
    
    f0 = plt.figure(figsize=(6,6))
    sns.distplot(bias01, label ="rho = 0.9")
    sns.distplot(bias02,label="rho = 0.5")
    sns.distplot(bias03,label="rho = 0.1")
    plt.legend()
    plt.xlabel("bias")
    plt.title('2,000 replications\n50 temperatures between (0,1)\n 1,000 draws at each temperature')
    f0.savefig('bc_MC_replications_R.pdf',bbox_inches='tight',dpi=100)
    
    

    		  	   		   	  			  	
    
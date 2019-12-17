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
# os.chdir('C:\\Users\\z5187692\\OneDrive - UNSW\\SDA\\SDA_project\\')
import bivariate_case as bc
import SDAutility as Sutil
import seaborn as sns


#%%

def bc_demonstration(mu,sigma,smin,smax,t_space,Method):
    log_phi0 = np.sum(np.log(smax-smin))
    expect_order0 = np.zeros(t_space.size)
    
    for t in range(t_space.size):
        bc0 = bc.Bivariate_Normal_Path_sample(smin,smax,t_space[t],\
                                              mu,sigma,1000,False,0,Method)
        bc0.get_expectation()
        expect_order0[t] = bc0.expectation    
    t0,final_t0 = Sutil.compute_integral_1st(expect_order0,t_space)
    result_order1 = final_t0 + log_phi0
    
    #print('estimation poly order 0:', result_order1)
    return result_order1




def test_01():
    mu = np.array([0.,0.])
    sigma = np.array([[1.,0.9],[0.9,1.]])
    obs = np.random.multivariate_normal(mu,sigma,size=100)
    smin = np.min(obs,axis=0)
    smax = np.max(obs,axis=0)
    num_t = 50
    t_space = np.log(np.logspace(start=0.0001,stop=1,base=np.e,num=num_t))

    est_botev = [bc_demonstration(mu,sigma,smin,smax,t_space,
                                  'Botev') for i in range(2000)]
    
    est_gibbs = [bc_demonstration(mu,sigma,smin,smax,t_space,'Gibbs')\
                   for i in range(2000)]
        
    est_qmc_rej = [bc_demonstration(mu,sigma,smin,smax,t_space,'QMC_rej')\
                   for i in range(2000)]

    est_rej  =   [bc_demonstration(mu,sigma,smin,smax,t_space,'Rej')\
                   for i in range(2000)]
    
    tv = bc.log_phi1(smin,smax,mu,sigma) 

    with open('test_01_result_all.txt', 'w') as f:
        for item in [est_botev,est_gibbs,est_qmc_rej,est_rej,tv,mu,sigma,smin,smax]:
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
    est_botev = [bc_demonstration(mu,sigma,smin,smax,t_space,
                                  'Botev') for i in range(2000)]
    est_gibbs = [bc_demonstration(mu,sigma,smin,smax,t_space,'Gibbs')\
                   for i in range(2000)]
    est_qmc_rej = [bc_demonstration(mu,sigma,smin,smax,t_space,'QMC_rej')\
                   for i in range(2000)]

    est_rej  =   [bc_demonstration(mu,sigma,smin,smax,t_space,'Rej')\
                   for i in range(2000)]
    
    tv = bc.log_phi1(smin,smax,mu,sigma) 

    with open('test_02_result_all.txt', 'w') as f:
        for item in [est_botev,est_gibbs,est_qmc_rej,est_rej,tv,mu,sigma,smin,smax]:
            f.write("%s\n" % item)
            
            
def test_03():
    mu = np.array([0.,0.])
    sigma = np.array([[1.,0.1],[0.1,1.]])
    obs = np.random.multivariate_normal(mu,sigma,size=100)
    smin = np.min(obs,axis=0)
    smax = np.max(obs,axis=0)
    num_t = 50
    t_space = np.log(np.logspace(start=0.0001,stop=1,base=np.e,num=num_t))
    est_botev = [bc_demonstration(mu,sigma,smin,smax,t_space,
                                  'Botev') for i in range(2000)]
    est_gibbs = [bc_demonstration(mu,sigma,smin,smax,t_space,'Gibbs')\
                   for i in range(2000)]
    est_qmc_rej = [bc_demonstration(mu,sigma,smin,smax,t_space,'QMC_rej')\
                   for i in range(2000)]

    est_rej  =   [bc_demonstration(mu,sigma,smin,smax,t_space,'Rej')\
                   for i in range(2000)]
    
    tv = bc.log_phi1(smin,smax,mu,sigma) 

    with open('test_03_result_all.txt', 'w') as f:
        for item in [est_botev,est_gibbs,est_qmc_rej,est_rej,tv,mu,sigma,smin,smax]:
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
    
def pre_process(fn):
    test = read_output(fn)
    test_res = np.fromstring(test,sep='\n')
    test_tv = test_res[-1]
    test_est = test_res[0:-1]
    return test_est,test_tv
    
    
#%% replication of 1000 times
if __name__ == "__main__":
    test_01() #loading output instead
    test_02() #loading output instead
    test_03() #loading output inteaad
    
    # note it is not quasi monte carlo 
    """
    os.chdir('C:\\Users\\z5187692\\OneDrive - UNSW\\SDA\\SDA_project\\output\\')
    test01_qmc_est, test01_qmc_tv = pre_process("test_01_result_qmc.txt")
    test02_qmc_est, test02_qmc_tv = pre_process("test_02_result_qmc.txt")
    test03_qmc_est, test03_qmc_tv = pre_process("test_03_result_qmc.txt")



    test01_mc_est, test01_mc_tv = pre_process("test_01_result.txt")
    test02_mc_est, test02_mc_tv = pre_process("test_02_result.txt")
    test03_mc_est, test03_mc_tv = pre_process("test_03_result.txt")

    ratio01_qmc = test01_qmc_est/test01_qmc_tv
    ratio02_qmc = test02_qmc_est/test02_qmc_tv
    ratio03_qmc = test03_qmc_est/test03_qmc_tv

    ratio01_mc = test01_mc_est/test01_mc_tv
    ratio02_mc = test02_mc_est/test02_mc_tv
    ratio03_mc = test03_mc_est/test03_mc_tv
     
    f0 = plt.figure(figsize=(6,6))
    sns.distplot(ratio01_qmc, label ="rho = 0.9")
    sns.distplot(ratio02_qmc,label="rho = 0.5")
    sns.distplot(ratio03_qmc,label="rho = 0.1")
    plt.legend()
    plt.xlabel("ratio (estimates/true value)")
    plt.title('2,000 replications\n50 temperatures between (0,1)\n 1,000 draws at each temperature')
    f0.savefig('bc_MC_replications_R_package.pdf',bbox_inches='tight',dpi=100)
    
    
    f1 = plt.figure(figsize=(6,6))
    sns.distplot(ratio01_mc, label ="rho = 0.9")
    sns.distplot(ratio02_mc,label="rho = 0.5")
    sns.distplot(ratio03_mc,label="rho = 0.1")
    plt.legend()
    plt.xlabel("bias")
    plt.title('2,000 replications\n50 temperatures between (0,1)\n 1,000 draws at each temperature')
    f1.savefig('bc_MC_replications_R_gibbs.pdf',bbox_inches='tight',dpi=100)
    
   
    f2, axes = plt.subplots(3, 1)
    sns.distplot(ratio01_mc, label ="rho = 0.9, Gibbs",ax=axes[0])
    sns.distplot(ratio01_qmc,label="rho = 0.9, Botev",ax=axes[0])
    axes[0].legend()
    sns.distplot(ratio02_mc, label ="rho = 0.5, Gibbs",ax=axes[1])
    sns.distplot(ratio02_qmc,label="rho = 0.5, Botev",ax=axes[1])
    axes[1].legend()
    sns.distplot(ratio03_mc, label ="rho = 0.1, Gibbs",ax=axes[2])
    sns.distplot(ratio03_qmc,label="rho = 0.1, Botev",ax=axes[2])
    axes[2].legend()
    f2.savefig('compare_between_Botev_Gibbs_sampling.pdf',bbox_inches='tight',dpi=100)
    """

    		  	   		   	  			  	
    
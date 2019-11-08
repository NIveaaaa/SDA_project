# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:43:54 2019

@author: z5187692
"""

xx = results.params[1:]

np.sum(grad_theta * xx,axis=1)

np.multiply(grad_theta,results.params[1:])

grad_theta*xx.T
        self.h_theta = np.multiply(grad_theta,xx)



uni = Univariate_Normal_Path_sample(smin,smax,0.5,mu,sigma,nsample=1000,use_CV=True)

plt.hist(uni.z)
uni.get_expection()
plt.hist(uni.h_theta)
plt.hist(uni.phi_theta)



        grad_theta = np.zeros(shape = (uni.nsample,2))
    
        grad_theta[:,0] = (uni.z - uni.mu)/np.power(uni.sigma_at_t,2) 
        grad_theta[:,1] = (-1/uni.sigma_at_t + np.power(uni.z-uni.mu,2)/
        np.power(uni.sigma_at_t,3))*1/np.sqrt(uni.t)
        
        x_theta = sm.add_constant(grad_theta)
        uni.get_phi_theta()
        model = sm.OLS(uni.phi_theta, x_theta)
        results = model.fit()
        print(results.summary())
        uni.get_expection()
        h =np.sum(np.multiply(grad_theta,results.params[1:]),axis=1)

        
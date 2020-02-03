"""
Some code for Yu Yang for the symbolic data analysis project.

This code implements the block-Poisson estimator. It assumes a sequence of estimators of A_m (see notes for notation). Your task is to implement A_m.
The code has options for generating a sobol sequence that may be used for implementing Quasi Monte Carlo integration. Your task is to make sure the "skip" argument works, see below
The code also implements a block proposal for U. This will be useful when you implement block pseudo marginal.
"""

import numpy as np
import scipy as sp
import numpy.random as npr
import scipy.stats as sps
import sobol_seq
import univariate_path_sampler as ups
import copy
#%%
npr.seed(100)
nSeq = 1 # This is used to keep track of where we are in the Sobol sequence we will generate.
M = 200 # Number of Monte Carlo / Quasi Monte Carlo numbers
M_BP = None # Number of symbols (= number of blocks in the block-Poisson)
Nobs_BP = np.repeat(100,30) # Number of observations per symbol
rho_BP = 0.8 # Correlation in the pseudo marginal
a_BP = 0.1 # The lower bound of the estimate of A_m.
theta = None # This will later be the parameter
S = None # This will later be S_min, S_max
#theta = np.array([0,1])
#S = np.array([[-2,2]]*M_BP)
d = 1 # dimension of the integral we are estimating with the Sobol sequence.
kwargs = {'M_BP' : M_BP,  'rho_BP' : rho_BP, 'd' : d, 'M' : M, 'approx_order': 1}
isQuasi = False # If True then Quasi random uniform numbers. Otherwise standard uniform numbers
temp = np.log(np.logspace(start=0.001,stop=1,num=200,base=np.e)) # temperature
#%%
def generate_Sobol(d, M):
    """
    Essentially a wrapper to the existing function but increases a global variable that keep tracks of where we are at the sequence.
    # TO YU: Note that the skip argument does not seem to have an effect (at least not in my version). Please make sure to update the package so it has an effect.
    # TO Matias: there is a bug on sobol_seq 0.1.2, the correct version should be installed from github source code. https://github.com/naught101/sobol_seq
    """
    global nSeq
    
    seq = sobol_seq.i4_sobol_generate(d, M, skip = 2 + M*nSeq)
    nSeq = nSeq + 1 # Increase global counter
    return seq

def init_random_numbers(isQuasi, **kwargs):
    """
    Initialize all random numbers for the block-Poisson estimator. 
    NOTE 1: These random variables are uniform (Quasi uniform if isQuasi is true, otherwise standard uniform). 
            They will later be use to generate normal random variates.
    NOTE 2: nSeq used here is a global variable that keep tracks of where in the Sobol sequence we are. Please note: See the "TO YU" note in generate_Sobol above. You need to make sure the skip argument works
    """
    M_BP, rho_BP, d, M = kwargs['M_BP'], kwargs['rho_BP'], kwargs['d'], kwargs['M']

    # The generated U has U[k][i] : i = 1, .. "Xi"_l and k = 1, ... , M_BP. Recall that Xi_l is Poisson(1). 
    if isQuasi:
        U = [[generate_Sobol(d, M) for item in range(sps.poisson.rvs(1))] for item in range(M_BP)]   
    else:
        # Plain pseudo-random numbers
        U = [[sps.uniform.rvs(size = (M, d)) for item in range(sps.poisson.rvs(1))] for item in range(M_BP)]

    kappa = np.ceil((1 - rho_BP)*M_BP).astype('int')
    
    # kappa is the number of blocks we will update in the pseudo-marginal MCMC. 

    return U, kappa

# Test init_random_numbers
#U, kappa = init_random_numbers(isQuasi, **kwargs)

def A_m_hat(theta, S, Nobs_BP,Random_nbrs):
    """
    Todo: Implement A_m_hat. This is the path-sampling estimator. The code inputs Random_nbrs (which may be Quasi or Standard) uniforms, which are used to compute the path sampling
          estimator. The code also inputs S (the rectangle) and theta (the parameter) 
          input: S (boundary points), Nobs_BP (num of observations) theta(parameter), nobs (number of observations per symbol)
          output: A_m_hat 
    """
    up =ups.univariate_path_sampler(S, Nobs_BP, temp, theta, Random_nbrs, **kwargs)
    return up.numerical_integral()
        
def log_abs_BP_estimator(theta, S, U, a_BP, **kwargs):
    """
    log of the absolute value of the block Poisson estimator. See notes.
    NOTE: theta is the parameter, S is the rectangle. I have not used them as I have not implemented A_m. This code assumes A_m is implemented. Your task is to implement A_m based on the uniform Quasi Random numbers (or standard uniform random numbers)
    """    
    M_BP, rho_BP, d, M = kwargs['M_BP'], kwargs['rho_BP'], kwargs['d'], kwargs['M']
    
    # A_m_hats has the same structure with U
    A_m_hats = copy.deepcopy(U)
  
    # Rewrite the following line
    #A_m_hats_minus_a_BP = [np.prod([(A_m_hat(theta, S, subset) - a_BP) for subset in item]) for item in U] # NOTE: If a list is empty (corresponding Xi_l is zero), then it automatically gets 1.0

    for r in range(len(U)):
        count = len(U[r])
        if count == 0:
            continue
        else:
            for j in range(count):
                A_m_hats[r][j] = A_m_hat(theta,S[r],Nobs_BP[r],np.asarray(U[r][j]).flatten())
    A_m_hats_minus_a_BP = [np.prod(np.asarray(item) - a_BP) for item in A_m_hats]
    # Check sign of estimator. Can be negative only if we have an odd number of negatives.
    isNegative = np.sum(np.array(A_m_hats_minus_a_BP) < 0) % 2 == 1
    anyNegative = np.sum(np.array(A_m_hats_minus_a_BP) < 0) > 0 # means that at least one negative (might result in positive if an odd number of them). This is not necessary to save, just doing it so you can keep track of how often we actually get a negative term
    log_abs_prods = np.log(np.abs(A_m_hats_minus_a_BP))    
    const = M_BP*(a_BP + 1)
    logEst = const + np.sum(log_abs_prods) # This is the log of the absolute value of the estimator.        
    
    # Rewrite the following line
    #lower_bound_sample = np.min([item for sublist in [[A_m_hat(theta, S, subset) for subset in item] for item in U if item] for item in sublist]) # Just to check what the actual lower bound of the terms is for these random numbers.
    lower_bound_sample = np.min([item for sublist in A_m_hats if sublist for item in sublist])
    return logEst, isNegative, anyNegative, lower_bound_sample


def PropU_given_currentU(U, kappa, isQuasi, **kwargs):
    """
    Block proposal of random numbers given the current random numbers U.
    This will be used in the pseudo-marginal algorithm. It will update a subset of the random numbers (kappa of the blocks) and keep the rest fixed
    """
    # To Yu: When you implement the pseudo marginal MCMC, you need to keep in mind that UProp here alternates U! So, if you reject in the MH ratio, you need to reset the U (PLEASE DO NOT FORGET THIS).
    # TODO later
    M_BP, rho_BP, d, M = kwargs['M_BP'], kwargs['rho_BP'], kwargs['d'], kwargs['M']

    U_Prop = U  

    BlocksToChange = npr.choice(M_BP, kappa, replace = False) # Which blocks to update

    for BlockToChange in BlocksToChange:
        if isQuasi:
            U_Prop[BlockToChange] = [generate_Sobol(d, M) for item in range(sps.poisson.rvs(1))]	
        else:
            U_Prop[BlockToChange] = [sps.uniform.rvs(size = (M, d)) for item in range(sps.poisson.rvs(1))]

    return U_Prop



# PropU_given_currentU(U, kappa, isQuasi, **kwargs)

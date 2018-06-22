#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy import sparse
from scipy.sparse.linalg import inv
from copy import deepcopy

sigma = 1.50               # utility parameter
delta = 0.10               # depreciation rate
beta = 0.95                # discount factor
alpha = 0.30               # capital elasticity of output
 
nbk = 1000                  # number of data points in the grid
crit = 1                   # convergence criterion
epsi = 1e-6                # convergence parameter

ks = ((1-beta*(1-delta)) / (alpha*beta))**(1/(alpha-1))

dev = 0.9                  # maximal deviation from steady state
kmin = (1-dev)*ks          # lower bound on the grid
kmax = (1+dev)*ks          # upper bound on the grid
devk = (kmax-kmin)/(nbk-1) # implied increment
kgrid = np.linspace(kmin,kmax,nbk)     # builds the grid

v = np.zeros((nbk))                # value function
kp0 = kgrid                          # initial guess on k(t+1)
dr = np.zeros((nbk))               # decision rule (will contain indices)
policy_rule = np.zeros((nbk,2))      # keeps policy rule together

iter_no = 0

while crit > epsi:
    iter_no += 1
    for i in range(0, nbk):
        # compute indexes for which consumption is positive
        tmp = kgrid[i]**alpha + (1 - delta) * kgrid[i] - kmin
        imax = min(math.floor(tmp/devk) + 1, nbk)
        
        # consumption and utility
        c = kgrid[i]**alpha + (1 - delta) * kgrid[i] - kgrid[0:imax]
        util = (np.power(c,(1-sigma)) - 1) / (1-sigma)
        
        # find new policy rule
        iteration = util + beta * v[0:imax]
        v[i] = np.amax(iteration, axis=0)
        dr[i] = np.argmax(iteration, axis=0)
        policy_rule[i] = [v[i], dr[i]]
    
    # decision rules
    kp = kgrid[policy_rule[:,1].astype(int)]
    c = np.power(kgrid, alpha) + (1-delta)*kgrid - kp
    
    # update the value
    util = (np.power(c, (1-sigma)) - 1) / (1 - sigma)
    Q = sparse.csr_matrix((nbk, nbk))
    
    for i in range(0, nbk):
        Q[i, dr[i]] = 1
    
    tv = sparse.linalg.inv(sparse.eye(nbk) - beta * Q) * util
    crit = max(abs(kp - kp0))
    print("\nIteration: ", iter_no)
    print("Crit: ", crit)
    
    kp0 = deepcopy(kp)
    v = deepcopy(tv)




















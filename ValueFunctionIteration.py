#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# author: Aiste Grusnyte
#

import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt

sigma = 1.5                 # utility parameter
delta = 0.1                 # depreciation rate
beta = 0.95                 # discount factor
alpha = 0.30                # capital elasticity of output

nbk = 1000                  # number of data points in the grid
crit = 1                    # convergence criterion
epsi = 1e-6                 # convergence parameter

ks = ((1 - beta * (1 - delta)) / (alpha * beta))**(1 / (alpha - 1))
dev = 0.99                  # maximal deviation from steady state
kmin = (1 - dev) * ks           # lower bound on the grid
kmax = (1 + dev) * ks           # upper bound on the grid

dk = (kmax - kmin) / (nbk - 1)    # implied increment
kgrid = np.linspace(kmin,kmax,nbk)  # builds the grid
v = np.zeros((nbk))         # value function
dr = np.zeros((nbk))        # decision rule (will contain indices)
tv = np.zeros((nbk))        # update value

value_function = np.zeros((nbk, 2))  # keeps value function together

# number of iterations
iter_no = 0

while crit > epsi:
    iter_no += 1
    for i in range(0, nbk):
        # computing indices for which consumption is positive
        tmp = kgrid[i]**alpha + (1 - delta) * kgrid[i] - kmin
        imax = min(math.floor(tmp / dk) + 1, nbk)
    
        # consumption and utility
        c = kgrid[i]**alpha + (1 - delta) * kgrid[i] - kgrid[0:imax]
        util = (np.power(c,(1 - sigma)) - 1) / (1 - sigma)
        
        # find value function
        value = util + beta * v[0:imax]
        
        tv[i] = np.amax(value)
        dr[i] = np.argmax(value, axis = 0)
        value_function[i] = [tv[i], dr[i]]
        
    crit = max(abs(tv - v)) # compute convergence criterion
    print("\nIteration: ", iter_no)
    print("Crit: ", crit)
    v = deepcopy(tv) # update the value function
    
kp = kgrid[value_function[:,1].astype(int)]
c = np.power(kgrid, alpha) + (1 - delta) * kgrid - kp
util = (np.power(c, (1 - sigma)) - 1) / (1 - sigma)

plt.plot(kgrid, v)
plt.title('Deterministic OGM')
plt.show()
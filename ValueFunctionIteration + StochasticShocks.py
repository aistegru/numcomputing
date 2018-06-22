#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from copy import deepcopy

sigma = 1.5                 # utility parameter
delta = 0.1                 # depreciation rate
beta = 0.95                 # discount factor
alpha = 0.30                # capital elasticity of output
rho = 0.80                  # persistence of the shock
se = 0.12                   # volatility of the shock

nbk = 1000                  # number of data points in the grid
nba = 2                     # number of values for the shock
crit = 1                    # convergence criterion
epsi = 1e-6                 # convergence parameter

# Discretization of the shock
p = (1+rho) / 2;
pi = [[p, 1-p],[1-p, p]]
a1 = np.exp(-se**2/(1-rho**2))
a2 = np.exp(se**2/(1-rho**2))
A = [a1, a2]

# Discretization of the state space
ks = ((1-beta*(1-delta)) / (alpha*beta))**(1/(alpha-1))
dev = 0.99                       # maximal deviation from steady state
kmin = (1-dev)*ks                # lower bound on the grid
kmax = (1+dev)*ks                # upper bound on the grid
kgrid = np.linspace(kmin,kmax,nbk)  # builds the grid
c = np.zeros((nbk,nba))          # consumption
util = np.zeros((nbk,nba))       # utility
v = np.zeros((nbk,nba))          # value function
tv = np.zeros((nbk,nba))         # update value

# value_function = np.zeros((nbk,nba))
value_function = []
dr = np.zeros(nbk) 

iter_no = 0

while crit > epsi:
    iter_no += 1
    value_function = []
    for i in range(0, nbk):
        for j in range(0, nba):
            c = A[j]*(kgrid[i]**alpha) + (1 - delta)*kgrid[i] - kgrid
            neg = np.where(c <= 0)
            c[neg] = math.nan
            
            util[:, j] = (np.power(c, (1-sigma)) - 1) / (1 - sigma)
            util[neg, j] = -1e12
          
        iteration = util + beta * (np.matmul(v[:], pi))
        tv[i] = np.amax(iteration, axis=0)
        dr[i] = np.argmax(iteration, axis=0)[0]
        value_function.append([[tv[i]], dr[i]])
          
    crit = np.amax(np.amax(abs((tv-v))))
    v = deepcopy(tv)
    print("\nIteration: ", iter_no)
    print("Crit: ", crit)
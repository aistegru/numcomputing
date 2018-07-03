#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# author: Aiste Grusnyte

import numpy as np
import math

T = 20                  # number of time periods
R = 1                   # gross interest rate
beta = 0.98             # subjective discount factor
delta = 1               # cost of working
y = 20                  # income from working
sigma = 1               # crra parameter, 1=log utility
n = 1000                # number of grid points for asset and (endogenous M grid)
amax = 50               # maximum value on asset grid
amin = 0.1              # minimum value on asset grid

# asset grid
agrid = np.linspace(amin, amax, n)

# defining function depending on sigma
def u(c):
    if sigma == 1:
        return math.log(c)
    else:
        return np.power(c, 1 - sigma) / 1 - sigma
    

def u_prime(c):
    if sigma == 1:
        return 1 / c
    else:
        return np.power(c, -sigma)
    
def u_prime1(c):
    if sigma == 1:
        return 1 / c
    else:
        return np.power(c, -1 / sigma)
    
# lists storing functions
        
Cw = np.zeros((T, n))        # consumption policy function for working individual
Cr = np.zeros((T, n))        # consumption policy function for retired individual

Mw = np.zeros((T, n))        # endogenous grid for working individuals
Mr = np.zeros((T, n))        # endogenous grid for retiring individuals

Vw = np.zeros((T, n))        # value function for working individual
Vr = np.zeros((T, n))        # value function for retired individual

V = np.zeros((T, n))         # value function
C = np.zeros((T, n))         # consumption policy function


# solve for t = T

Cw[T - 1] = agrid
Cr[T - 1] = agrid

Mw[T - 1] = agrid
Mr[T - 1] = agrid

Vw[T - 1] = Cw[T - 1]
Vr[T - 1] = Cr[T - 1]

V[T - 1] = np.maximum(Vw[T - 1], Vr[T - 1])
C[T - 1] = Cw[T - 1]
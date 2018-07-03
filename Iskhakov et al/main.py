#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# author: Aiste Grusnyte

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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
        return np.log(c)
    else:
        return np.power(c, 1 - sigma) / (1 - sigma)
    
def u_prime(c):
    if sigma == 1:
        return np.divide(1., c)
    else:
        return np.power(c, -sigma)
    
def u_prime1(c):
    if sigma == 1:
        return np.divide(1., c)
    else:
        return np.power(c, -1 / sigma)
    
def interpolated(x, y, query_points):
    f = interp1d(x, y, kind='linear', fill_value = 'extrapolate')
    return f(query_points)
    
# lists storing functions
        
Cw = np.zeros((T, n))        # consumption policy function for working individual
Cr = np.zeros((T, n))        # consumption policy function for retired individual

Mw = np.zeros((T, n))        # endogenous grid for working individuals
Mr = np.zeros((T, n))        # endogenous grid for retiring individuals

Vw = np.zeros((T, n))        # value function for working individual
Vr = np.zeros((T, n))        # value function for retired individual

V = np.zeros((T, n))         # value function
C = np.zeros((T, n))         # consumption policy function

'''
SOLVING FOR t = T
'''
Cw[T - 1,:] = agrid
Cr[T - 1,:] = agrid

Mw[T - 1,:] = agrid
Mr[T - 1,:] = agrid

Vw[T - 1,:] = u(Cw[T - 1])
Vr[T - 1,:] = u(Cr[T - 1])

V[T - 1, :] = np.maximum(Vw[T - 1, :], Vr[T - 1, :])
C[T - 1, :] = Cw[T - 1, :]

'''
SOLVING FOR t = T - 1
'''

# for retiree
rhs = (beta * R) * u_prime(R * agrid)
c = u_prime1(rhs)
m = c + agrid

Cr[T - 2,:] = c
Mr[T - 2,:] = m
Vr[T - 2,:] = u(Cr[T - 2,:]) + beta * interpolated(Mr[T - 1,:], Vr[T - 1,:], R * (Mr[T - 2,:] - Cr[T - 2,:]))

# for worker
rhs = (beta * R) * u_prime(R * agrid + y)
c = u_prime1(rhs)
m = c + agrid

Cw[T - 2,:] = c
Mw[T - 2,:] = m
Vw[T - 2,:] = u(Cw[T - 2,:]) - delta + beta * interpolated(Mw[T - 1,:], Vw[T - 1,:], R * (Mw[T - 2,:] - Cw[T - 2,:]) + y)

Vw[T - 2,:] = interpolated(Mw[T - 2,:], Vw[T - 2,:], agrid)
Vr[T - 2,:] = interpolated(Mr[T - 2,:], Vr[T - 2,:], agrid)

# where functions cut
index = np.where(Vw[T-2,:] - Vr[T-2,:] < 0)
test = Vw[T-2] - Vr[T-2]

# value function
V[T - 2,:] = np.maximum(Vr[T - 2,:], Vw[T - 2,:])

# consumption policy
Cw[T - 2,:] = interpolated(Mw[T - 2,:], Cw[T - 2,:], agrid)
Cr[T - 2,:] = interpolated(Mr[T - 2,:], Cr[T - 2,:], agrid)

C[T - 2, :] = Cr[T - 2, :]
C[T - 2, index] = Cw[T - 2, index]

'''
SOLVING FOR t = T - 2
'''

# for retiree
rhs = (beta * R) * u_prime(R * agrid)
c = u_prime1(rhs)
m = c + agrid

Cr[T - 3, :] = c
Mr[T - 3, :] = m
Vr[T - 3, :] = u(Cr[T - 3, :]) + beta * interpolated(Mr[T - 2, :], Vr[T - 2, :], R * (Mr[T - 3, :] - Cr[T - 3, :]))

# plotting the graphs for figure(a)
M = [28, 29, 30.5826, 31, 32]
C = np.linspace(10, 30, 1000)
for i in range (5):
    V[i,:] = u(C) + beta * interpolated(agrid, V[T - 2, :], R * (M[i] - C) + y)
    plt.plot(C, V[i,:])
    
plt.plot(C, 30.4382)
plt.show()       

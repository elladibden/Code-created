#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 08:58:30 2023

@author: elladibden
"""
import numpy as np
import matplotlib.pyplot as plt
import math


t = np.linspace(0,250,50)
x = np.linspace(-100,100)
x0 =-80
v = 0.5
sigma0 = 10
sigma = sigma0*np.sqrt(1+((t**2)/(sigma0**4)))
Pxt = (1/(sigma*np.sqrt(math.pi)))*np.exp(-((x-x0-(v*t))**2)/(sigma**2))

plt.plot(t,Pxt)
plt.xlabel('t')
plt.ylabel('P')
plt.legend(('P(x,t)'), loc = 'center right')

def schrodingerfunc(n):
    N = 500              # Number of grid points
    L = 200              # System extends from -L/2 to L/2
    h = L/(N-1)          # Grid size
    tau = 0.1            # Time step
    x = h*np.arange(0, N) - L/2  # x values
    ham = np.zeros([N, N])
    coeff = -1/(2*h**2)
    for k in range(1, N-1):     # k values from 2 to (N-2)
        ham[k, k-1] = ham[k, k+1] = coeff
        ham[k, k] = -2*coeff  # Set interior rows

# Periodic boundary conditions
    ham[0, N-1] = ham[0, 1] = ham[N-1, N-2] = ham[N-1, 0] = coeff
    ham[0, 0] = ham[N-1, N-1] = -2*coeff

# Compute the Crank-Nicolson matrix
    mat = np.eye(N) + 1j*0.5*tau*ham
    dCN = np.linalg.inv(mat) @ np.conjugate(mat)

# Initialize wavefunction
    x0 = -80          # Location of the centre of the wavepacket
    velocity = 0.5    # Average velocity of the packet
    k0 = velocity     # Average wavenumber
    sigma0 = 10       # Standard deviation of the wavefunction

    if n == 1:        # Gaussian wavepacket
        norm_psi = 1/(np.sqrt(sigma0*np.sqrt(np.pi)))  # Normalization
        psi = norm_psi * np.exp(1j*k0*x) * np.exp(-(x-x0)**2/(2*sigma0**2))
    elif n == 2:      # Wavepacket with abs x
        norm_psi = 1/(np.sqrt(2*sigma0))  # Normalization
        psi = norm_psi * np.exp(1j*k0*x) * np.exp(-np.abs(x-x0)/(2*sigma0))

    max_iter = 2500   # Maximum number of iterations
    plot_iter = 500   # Iterations to record
    iplot = 1         # Initialise ploting index

    ncols = max_iter//plot_iter  # // is floor division
    prob = np.zeros([N, ncols+1])  # Array to contain the results

# Multplying a complex number by its conjugate gives a real number, but the
# result here is a complex number with zero imaginary part. Take real part.
    prob[:, 0] = np.real(psi*np.conjugate(psi))   # Record initial condition

# Do the main calculation
    for iter in range(max_iter):
        psi = dCN @ psi  # Crank-Nicolson scheme
        if (iter+1) % plot_iter < 1:  # Remainder after division
            prob[:, iplot] = np.real(psi*np.conjugate(psi))
            iplot += 1
            
            return(x,prob)

x1,prob1= schrodingerfunc(1)
plt.figure()
plt.plot(x1,prob1)
plt.xlabel('Position')
plt.ylabel('Probability')

x2,prob2 = schrodingerfunc(2)

#%%

fig = plt.figure(figsize=(12,10))
plt.subplots_adjust(wspace=0.4, hspace=0.4, top=0.85)
fig.suptitle('results for gaussian and non-gaussian wavepacket', fontsize=18)
ax1 = fig.add_subplot(2, 2, 1) 
ax1.plot(t,Pxt)
ax1.set_title('gaussian with function P(x,t)')
ax2 = fig.add_subplot(2, 2, 2) 
axc = ax2.plot(x1,prob1)
ax2.set_title('gaussian with schrodinger')
ax3 = fig.add_subplot(2, 2, 3) 
ax3.plot(x2,prob2)
ax3.set_title('non gaussian with schrodinger')
plt.savefig('examq3ciii.jpg')


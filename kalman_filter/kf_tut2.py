#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:39:13 2024

@author: gitartha

Estimating the position of a accelarating vehicle in 1D
using predictions of the noisy process and uncertain measurements
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

"########################## Parameters ##############################"
process_noise_std = 10
measurement_noise_std = 10

pos_guess_std = 10

acc = 50
x_init = np.array([[0.], [0.]])
# State vector containing position and velocity

df = 0.01 # For sensitivity calculations
dt = 0.1
Tend = 10 

Case = 1
"######################### Initialize ############################"
T = 0.0
iteration = 0

acc_true = acc

x_true = x_init
x = x_init

F = np.array([[1, dt], [0, 1]])
g = np.array([[0.5 * dt**2], [dt]])
H = np.array([[1.,0]])

pos_var = pos_guess_std ** 2
vel_var = pos_var / dt**2
acc_var = process_noise_std**2

P = np.array([[pos_var, pos_var/dt], [pos_var/dt, vel_var]])
Q = np.array([[dt**2, dt**3], [dt**3, dt**4]]) * acc_var
R = np.array([[measurement_noise_std**2]])

I = np.identity(P.shape[0])

pos_estimate = [x[0][0]]
pos_measured = [x_init[0][0]]
pos_true = [x_init[0][0]]
vel_true = [x_init[1][0]]
vel_estimate = [x[1][0]]
time = [0]

sens = np.zeros(F.shape)
sens_save_00 = [0]
sens_save_01 = [0]
sens_save_10 = [0]
sens_save_11 = [0]

"######################## Main loop ################################"

while T < Tend:
    
    T += dt
    iteration += 1
    
    if Case == 2:
        acc = abs(acc) * (-1) ** int(T)
    if Case == 3:
        if T>Tend/2:
            acc = -abs(acc)
        
    
    acc_true = acc + np.random.normal(0,process_noise_std)
    x_true = F @ x_true + acc_true * g
    
    z = H @ x_true + np.random.normal(0,measurement_noise_std)
    
    "-------------- Sensitivity calculation -----------------"
    # using finite difference to compute sensitivity of 
    # Cost function sqrt((z - Hx)**2) wrt elements of F
    
    
    '''
    
    if iteration == 20:
        cost = []
        deltaF = np.arange(-2,2,0.01)
        for dF0 in deltaF:
            F0 = F*1 
            F0[1][1] += dF0
            cost.append(np.sqrt((z - H @ (F0 @ x + acc*g))**2)[0][0])
        
        plt.figure(dpi=150)
        plt.plot(deltaF, cost)
        plt.title("Graph of cost function wrt F[1][1]")
        plt.xlabel("dF[i][j]")
        plt.show()
        
        sys.exit()
    
    '''
    
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            Fp = 1*F
            Fm = 1*F
            Fp[i][j] = Fp[i][j] + df
            Fm[i][j] = Fm[i][j] - df
            
            costp = np.sqrt((z - H @ (Fp @ x + acc*g))**2)
            costm = np.sqrt((z - H @ (Fm @ x + acc*g))**2)
            
            sens[i][j] = 0.5*(costp[0][0] - costm[0][0])/df
            
    sens_save_00.append(sens[0][0])
    sens_save_01.append(sens[0][1])
    sens_save_10.append(sens[1][0])
    sens_save_11.append(sens[1][1])
     
    
    "------------------- State Update -----------------------"
    L = H @ P @ np.transpose(H) + R
    K = P @ np.transpose(H) @ np.linalg.inv(L)
    # PATCH HERE
     
    x = x + K @ (z - H @ x)
    P = (I - K @ H) @ P
    
    "-------------------- Save Data --------------------------"
    pos_estimate.append(x[0][0])
    vel_estimate.append(x[1][0])
    pos_true.append(x_true[0][0])
    vel_true.append(x_true[1][0])
    pos_measured.append(z[0][0])
    time.append(T)
    
    "------------------- State Predict -----------------------"
    
    x = F @ x + acc * g
    P = F @ P @ np.transpose(F) + Q

#################################################################
############################### PLOT ############################

plt.figure(dpi=150)

plt.plot(time, pos_estimate, label="Estimate", color='r')
plt.plot(time, pos_true, label="True", color='b')
plt.plot(time, pos_measured, '-.', label="Measured", color='g')

plt.title("Graph of Position vs. Time")
plt.legend()
plt.show()

plt.figure(dpi=150)

plt.plot(time, vel_estimate, label="Estimate", color='r')
plt.plot(time, vel_true, label="True", color='b')

plt.title("Graph of velocity vs. Time")
plt.legend()
plt.show()

plt.figure(dpi=150)
 
plt.plot(time, sens_save_00, label="F[0][0]", color='r')
plt.plot(time, sens_save_01, label="F[0][1]", color='b')
plt.plot(time, sens_save_10, label="F[1][0]", color='g')
plt.plot(time, sens_save_11, label="F[1][1]", color='y')

plt.title("Sensitivity w.r.t. F vs time")
plt.legend()
plt.show()


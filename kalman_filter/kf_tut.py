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

"########################## Parameters ##############################"
process_noise_std = 1
measurement_noise_std = 1
pos_guess_std = 10

Acc = 50
Vinit = 0.0
Xinit = 0.0

dt = 0.01
Tend = 40


"######################## Initial values ############################"
T = 0.0

acc_true = Acc
vel_true = Vinit
pos_true = Xinit

pos_predict = 0.0   # initial guess for position
vel_predict = 0.0   # initial guess for velocity
acc_predict = Acc   # initial guess for accelaration

pos_var = pos_guess_std**2    # variance of the position estimate
#vel_var = 0.0
#acc_var = 0.0

measure_var = measurement_noise_std**2  # variance of measurements uncertainity 
process_var = process_noise_std**2  # variance of process noise

pos_estimate = pos_predict
vel_estimate = vel_predict
acc_estimate = acc_predict

pos_estimate_save = []
pos_measured_save = []
pos_predict_save = []
pos_true_save = []

vel_estimate_save = []
vel_predict_save = []
vel_true_save = []

acc_estimate_save = []
acc_predict_save = []
acc_true_save = []

"######################## Main loop ################################"

while T < Tend:
    
    T += dt
    
    Acc = abs(Acc) * (-1) ** int(T)
    
    pos_predict_save.append(pos_predict)
    vel_predict_save.append(vel_predict)
    acc_predict_save.append(acc_predict)
    
    acc_true = Acc + np.random.normal(0,process_noise_std)
    vel_true = vel_true + dt * acc_true
    pos_true = pos_true + dt * vel_true
    
    pos_measured = pos_true + np.random.normal(0,measurement_noise_std)
    
    "--------------------------------------------------------"
    last_pos_estimate = pos_estimate
    last_vel_estimate = vel_estimate
    
    "------------------- State Update -----------------------"
    
    K = pos_var / (pos_var + measure_var)
    
    pos_estimate = pos_predict + K*(pos_measured - pos_predict)
    
    pos_var = (1 - K) * pos_var
    
    vel_estimate = (pos_estimate - last_pos_estimate)/dt
    acc_estimate = (vel_estimate - last_vel_estimate)/dt
    
    "------------------- State Predict -----------------------"
    
    acc_predict = acc_estimate
    vel_predict = vel_estimate + dt * acc_estimate
    pos_predict = pos_estimate + dt * vel_estimate
    
    #acc_var = acc_var + process_var
    #vel_var = vel_var + dt**2 * acc_var
    #pos_var = pos_var + dt**3 * vel_var
    
    "-------------------- Save Data --------------------------"
    
    pos_estimate_save.append(pos_estimate)
    pos_measured_save.append(pos_measured)
    pos_true_save.append(pos_true)

    vel_estimate_save.append(vel_estimate)
    vel_true_save.append(vel_true)

    acc_estimate_save.append(acc_estimate)
    acc_true_save.append(acc_true)

#################################################################
############################### PLOT ############################

x = range(len(pos_estimate_save))
#marker = ['-o', '-s', 'v', '-*']
marker = ['-', '-.', '--', '-']
plt.figure(dpi=150)

plt.plot(x, pos_estimate_save, marker[0], label="Estimate")
plt.plot(x, pos_predict_save, marker[1], label="Predicted")
plt.plot(x, pos_true_save, marker[2], label="True")
plt.plot(x, pos_measured_save, marker[3], label="Measured")

plt.title("Graph of Position vs. Time")
plt.legend()
plt.show()

""
plt.figure(dpi=150)

plt.plot(x, vel_estimate_save, marker[0], label="Estimate")
#plt.plot(x, vel_predict_save, marker[1], label="Predicted")
plt.plot(x, vel_true_save, marker[2], label="True")

plt.title("Graph of velocity vs. Time")
plt.legend()
plt.show()

plt.figure(dpi=150)

plt.loglog(x, acc_estimate_save, marker[0], label="Estimate")
#plt.plot(x, acc_predict_save, marker[1], label="Predicted")
plt.plot(x, acc_true_save, marker[2], label="True")
plt.yscale('symlog')
plt.xscale('linear')

plt.title("Graph of Accelaration vs. Time")
plt.legend()
plt.show()

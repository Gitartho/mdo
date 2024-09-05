#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:39:13 2024

@author: gitartha

Estimating the position of a accelarating vehicle in 1D
using predictions of the noisy process and uncertain measurements
"""
import numy as np

"########################## Parameters ##############################"
process_noise_std = 10
measurement_noise_std = 10

acc = 50
Vinit = 0.0
Xinit = 0.0

dt = 0.01
Tend = 10


"######################## Initial values ############################"
T = 0.0
vel_true = 0.0
pos_true = 0.0

pos_predict = 0.0   # initial guess for position
pos_var = (5)**2    # variance of the position estimate
measure_var = measurement_noise_std**2  # variance of measurements uncertainity 
process_var = process_noise_std**2  # variance of process noise

pos_estimate = pos_predict
vel_estimate = 0.0


"######################## Main loop ################################"

while T < Tend:
    
    acc_true = acc + np.random.normal(0,process_noise_std)
    vel_true = vel_true + dt * acc_true
    pos_true = pos_true + dt * vel_true
    
    pos_measured = pos_true + np.random(0,measurement_noise_std)
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


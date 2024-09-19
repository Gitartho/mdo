"""
Created on Tue Sep 17 22:10:16 2024

@author: Gitartha

Estimating the position of a accelarating vehicle in 1D
using predictions of the noisy process and uncertain measurements
And performing optimization
"""
import numpy as np
import matplotlib.pyplot as plt

"########################## Parameters ##############################"
process_noise_std = 10
measurement_noise_std = 10

pos_guess_std = 10

acc = 50
x_init = np.array([[0.], [0.]])
# State vector containing position and velocity

dt = 0.1
Tend = 10.0

Case = 1

"################### Optimization parameters ########################"

pertb_time = 5.0 # time at which we want to perturb the F matrix
pertb_i = 0
pertb_j = 0 # indices at which we want to purturn the F matrix

cost_X_limit = 0.2 # cost function will be plotted in this range [-x, x]
cost_stepsize = 0.01 

cost_save = []
DF = np.arange(-cost_X_limit, cost_X_limit+cost_stepsize, cost_stepsize)

"------------------------------------------------------------------------------"
"Generate the random data in advanced for consistancy bw optimization iterations"

acc_noise = []
z_noise = []

for temp in np.arange(0, Tend+dt, dt):
    acc_noise.append(np.random.normal(0,process_noise_std))
    z_noise.append(np.random.normal(0,measurement_noise_std))  


def KF():
    
    global acc, acc_noise, z_noise, x_init, dt, Tend, Case 
    global pos_guess_std, process_noise_std, measurement_noise_std

    "--------------------- Initialize ------------------------"
    time_iter = 0
    T = 0.0
    
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
    
    "####################### Main Time Loop #####################"
    while T < Tend:
        
        T += dt
        time_iter += 1
        
        if Case == 2:
            acc = abs(acc) * (-1) ** int(T)
        if Case == 3:
            if T>Tend/2:
                acc = -abs(acc)
        
        acc_true = acc + acc_noise[time_iter-1]
        x_true = F @ x_true + acc_true * g
        
        z = H @ x_true + z_noise[time_iter-1]
        
        "----------------- Compute Cost Function ----------------"
        
        F0 = F*1
        if(time_iter == pertb_time/dt):
            F0[pertb_i][pertb_j] += dF

        cost += np.sqrt((z - H @ (F0 @ x + acc*g))**2)[0][0] / (2*cost_X_limit/cost_stepsize)
  
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
    
    cost_save.append(cost)
  
 
#################################################################
############################### PLOT ############################

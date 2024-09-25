"""
Created on Tue Sep 17 22:10:16 2024

@author: Gitartha

Estimating the position of a accelarating vehicle in 1D
using predictions of the noisy process and uncertain measurements
And performing optimization
"""
import numpy as np
import matplotlib.pyplot as plt
import niceplots
from scipy.optimize import minimize
import niceplots

plt.style.use(niceplots.get_style())

########################## Parameters ##############################
process_noise_std = 10
measurement_noise_std = 10

pos_guess_std = 10

acc = 50
x_init = np.array([[0.], [0.]])
# State vector containing position and velocity

dt = 0.1
Tend = 10.0

Case = 2
plot = True

################### Optimization parameters ########################

pertb_time = 5.0
pertb_i = 0
pertb_j = 0

cost_X_limit = 40 # cost function will be plotted in this range [-x, x]
cost_step = 0.1
opt_iter_lim = 10 # Optimization iterations


u_guess = 0.0

"---------------------------------------------------------------------"
"Generate the random data in advanced for consistancy"

acc_noise = []
z_noise = []

for temp in np.arange(0, Tend+dt, dt):
    acc_noise.append(np.random.normal(0,process_noise_std))
    z_noise.append(np.random.normal(0,measurement_noise_std))  


#===================================================================#
##################### Kalman Filter Function ########################
def KF(FF):
    
    global acc, acc_noise, z_noise, x_init, dt, Tend, Case, plot
    global pos_guess_std, process_noise_std, measurement_noise_std
    
    cost = 0.0

    "--------------------- Initialize ------------------------"
    time_iter = 0
    T = 0.0
    
    acc_true = acc

    x_true = x_init
    x = x_init

    Ftrue = np.array([[1, dt], [0, 1]])
    g = np.array([[0.5 * dt**2], [dt]])
    H = np.array([[1.,0]])

    pos_var = pos_guess_std ** 2
    vel_var = pos_var / dt**2
    acc_var = process_noise_std**2

    # P = np.array([[pos_var, pos_var/dt], [pos_var/dt, vel_var]])
    P = np.eye(2)  # Initial error covariance matrix
    # Q = np.array([[dt**2, dt**3], [dt**3, dt**4]]) * acc_var
    Q = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * process_noise_std ** 2  # Process noise covariance
    R = np.array([[measurement_noise_std**2]])

    I = np.identity(P.shape[0])
    
    pos_predict = [x[0][0]]
    pos_estimate = [x[0][0]]
    pos_measured = [x_init[0][0]]
    pos_true = [x_init[0][0]]
    vel_true = [x_init[1][0]]
    vel_estimate = [x[1][0]]
    
    time = [0]
    
    ####################### Main Time Loop #####################
    while T < Tend:
        
        T += dt
        time_iter += 1
        
        if Case == 2:
            acc = abs(acc) * (-1) ** int(T)
        if Case == 3:
            if T>Tend/2:
                acc = -abs(acc)
        
        acc_true = acc + acc_noise[time_iter-1]
        x_true = Ftrue @ x_true + acc_true * g
        F = FF[time_iter-1]*1
        
        z = H @ x_true + z_noise[time_iter-1]
  
        "------------------- State Update -----------------------"
        L = H @ P @ np.transpose(H) + R
        K = P @ np.transpose(H) @ np.linalg.inv(L)
        # PATCH HERE
         
        x = x + K @ (z - H @ x)
        P = (I - K @ H) @ P
        
        "----------------- Compute Cost Function ----------------"

        cost += np.sqrt((z - H @ x)**2)[0][0]
        
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
        
        pos_predict.append(x[0][0])
        
    if plot:
        
        fig, ax = plt.subplots(dpi=400)

        ax.plot(time, pos_estimate, label="Estimate", color='r')
        ax.plot(time, pos_true, label="True", color='b')
        ax.plot(time, pos_measured, '-.', label="Measured", color='g')
        # ax.plot(time, pos_predict, '--', label='predicted', color='black')

        ax.set_xlabel("time")
        ax.set_ylabel("x", rotation="horizontal", ha="right")
        niceplots.adjust_spines(ax)
        # niceplots.label_line_ends(ax)
        plt.legend()

        plot = False
    
    return cost/(Tend/dt)


#===================================================================#
##################### Optimization Section ##########################

def objective(u):
    
    global Tend, dt
    
    FF = []
    for i in range(int(Tend/dt)+1):
        F = np.array([[1, dt], [0, 1]])
        if i == pertb_time/dt: 
            F[pertb_i][pertb_j] = u[0]
        FF.append(F)
        
    return KF(FF)

if 0:
    U = np.arange(-cost_X_limit, cost_X_limit, cost_step)
    cost_func = []
    for u in U:
        cost_func.append(objective([u]))
    
    plt.style.use(niceplots.get_style())
    fig, ax = plt.subplots(dpi=400)
    ax.plot(U, cost_func, color='r')
    ax.set_xlim(-cost_X_limit, cost_X_limit)
    ax.set_xlabel(f'F[{pertb_i}][{pertb_j}]')
    ax.set_ylabel("Cost", rotation="horizontal", ha="right", va="center")
    # niceplots.adjust_spines(ax)
    # niceplots.label_line_ends(ax)
 

if 0:
    u_bound = [[-cost_X_limit, cost_X_limit]]
    u0 = [u_guess]
    
    solution = minimize(objective, u0, method='SLSQP', bounds=u_bound)
    print(solution)
    
    plot = True
    objective(solution.x)
    
if 1:
    plot = True
    objective([1])
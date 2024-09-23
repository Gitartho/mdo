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
plot = False

################### Optimization parameters ########################

pertb_time = 5.0
pertb_i = 0
pertb_j = 0

cost_X_limit = 10 # cost function will be plotted in this range [-x, x]
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
        
    if plot:
        
        fig, ax = plt.subplots()
        
        ax.figure(dpi=400)

        ax.plot(time, pos_estimate, label="Estimate", color='r')
        ax.plot(time, pos_true, label="True", color='b')
        ax.plot(time, pos_measured, '-.', label="Measured", color='g')

        ax.set_xlabel("t")
        ax.set_ylabel("$\sigma$", rotation="horizontal", ha="right")
        niceplots.adjust_spines(ax)
        niceplots.label_line_ends(ax)

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

U = np.arange(-cost_X_limit, cost_X_limit, cost_step)
cost_func = []
for u in U:
    cost_func.append(objective([u]))

plt.style.use(niceplots.get_style())
fig, ax = plt.subplots()
ax.plot(U, cost_func, color='r')
ax.set_xlim(-cost_X_limit, cost_X_limit)
ax.set_xlabel(f'F[{pertb_i}][{pertb_j}]')
ax.set_ylabel("Cost Function", rotation="vertical", ha="right", va="center")
# niceplots.adjust_spines(ax)
# niceplots.label_line_ends(ax)
 

    

        
# u_bound = [[-cost_X_limit, cost_X_limit]]
# u0 = [u_guess]

# solution = minimize(objective, u0, method='SLSQP', bounds=u_bound)
# print(solution)

# plot = True
# objective(solution.x)

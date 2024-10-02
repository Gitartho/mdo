import numpy as np
import matplotlib.pyplot as plt

def system(x):
    # Constants
    g = 9.81  # Acceleration due to gravity (m/s^2)
    thrust = 30.0  # Thrust force (N)
    mass = 2.0  # Rocket mass (kg)
    burn_time = 10.0  # Burn time of the engine (s)
    dt = 1  # Time step (s)
    
    # Initial conditions
    true_velocity = 0.0  # True initial velocity (m/s)
    true_altitude = 0.0  # True initial altitude (m)
    measured_altitude = 0.0  # Initial measured altitude (m)
    measurement_noise_std = 5.0
    estimated_state = np.array([[0.0], [0.0]])  # Initial estimated state [altitude, velocity]
    
    F = np.array([[1, dt], [0, 1]])  # State transition matrix
    H = np.array([[1, 0]])  # Observation matrix
    f = np.array([[0.5*dt**2], [dt]])
    
    
    # Lists to store time, true altitude, true velocity, measured altitude, estimated altitude, and estimated velocity for plotting
    state = []
    objective = []
    
    # Time loop
    time = 0.0
    while time <= 20.0:
        
        # True state update
        true_acceleration = thrust / mass - g 
        guess_acceleration = 1.0 * true_acceleration + x #[int(time/dt)]
        true_acceleration += np.random.normal(0, 5.0)
        "Say the acceleration value has large uncertainity due to friction etc."
    
        true_velocity += true_acceleration * dt
        true_altitude += true_velocity * dt + 0.5 * guess_acceleration * dt**2 
    
        # Simulate noisy measurement
        measured_altitude = true_altitude + np.random.normal(0, measurement_noise_std)
        
        "=================================================================="         
        estimated_state = F @ estimated_state + true_acceleration * f
        
        state.append(estimated_state)
        objective.append(measured_altitude - H @ estimated_state)
    
        # Increment time
        time += dt
        
    return state, np.linalg.norm(objective)

def residual(x, u):
    
    dt = 1
    
    F = np.array([[1, dt], [0, 1]])  # State transition matrix
    f = np.array([[0.5*dt**2], [dt]])
    
    x_im1 = np.array([[0.0], [0.0]])
    x_i = np.array([[0.0], [0.0]])
    r = np.zeros([20, 2])
    
    for i in range(20):
        x_i[0] = x[i][0]
        x_i[0] = x[i][1]
        res = x_i - x_im1 - dt * F @ x_im1 - u * f
        r[i] = res.T
        x_im1[0] = x[i][0]
        x_im1[0] = x[i][1]
        
    return r

x = np.tile([0.0, 0.0], (20,1))
u = 5.19

r = residual(x,u)
print(r)

        
    
    
import numpy as np
import jax
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)
thrust = 30.0  # Thrust force (N)
mass = 2.0  # Rocket mass (kg)
burn_time = 10.0  # Burn time of the engine (s)
dt = 0.01  # Time step (s)

# Initial conditions
true_velocity = 0.0  # True initial velocity (m/s)
true_altitude = 0.0  # True initial altitude (m)
measured_altitude = 0.0  # Initial measured altitude (m)
process_noise_std = 0.1  # Standard deviation of process noise
measurement_noise_std = 5.0  # Standard deviation of measurement noise

# Kalman filter initial conditions
estimated_state = np.array([[0.0], [0.0]])  # Initial estimated state [altitude, velocity]
P = np.eye(2)  # Initial error covariance matrix
F = np.array([[1, dt], [0, 1]])  # State transition matrix
H = np.array([[1, 0]])  # Observation matrix
R = np.array([[measurement_noise_std ** 2]])  # Measurement noise covariance
Q = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * process_noise_std ** 2  # Process noise covariance

# Lists to store time, true altitude, true velocity, measured altitude, estimated altitude, and estimated velocity for plotting
time_list = []
true_altitude_list = []
true_velocity_list = []
measured_altitude_list = []
estimated_altitude_list = []
estimated_velocity_list = []

# Time loop
time = 0.0
while time <= burn_time:
    # True state update
    true_acceleration = thrust / mass - g
    true_velocity += true_acceleration * dt
    true_altitude += true_velocity * dt

    # Simulate noisy measurement
    measured_altitude = true_altitude + np.random.normal(0, measurement_noise_std)

    # Kalman filter prediction step
    estimated_state = F @ estimated_state
    P = F @ P @ F.T + Q

    # Kalman filter update step
    y = measured_altitude - H @ estimated_state  # Measurement residual
    S = H @ P @ H.T + R  # Residual covariance
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
    estimated_state = estimated_state + K @ y
    P = (np.eye(2) - K @ H) @ P

    # Store values for plotting
    time_list.append(time)
    true_altitude_list.append(true_altitude)
    true_velocity_list.append(true_velocity)
    measured_altitude_list.append(measured_altitude)
    estimated_altitude_list.append(estimated_state[0, 0])
    estimated_velocity_list.append(estimated_state[1, 0])

    # Increment time
    time += dt

# Continue simulation without thrust after burn time
while time <= 20.0:
    # True state update
    true_acceleration = -g
    true_velocity += true_acceleration * dt
    true_altitude += true_velocity * dt

    # Simulate noisy measurement
    measured_altitude = true_altitude + np.random.normal(0, measurement_noise_std)

    # Kalman filter prediction step
    estimated_state = F @ estimated_state
    P = F @ P @ F.T + Q

    # Kalman filter update step
    y = measured_altitude - H @ estimated_state  # Measurement residual
    S = H @ P @ H.T + R  # Residual covariance
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
    estimated_state = estimated_state + K @ y
    P = (np.eye(2) - K @ H) @ P

    print("P", P)

    # Store values for plotting
    time_list.append(time)
    true_altitude_list.append(true_altitude)
    true_velocity_list.append(true_velocity)
    measured_altitude_list.append(measured_altitude)
    estimated_altitude_list.append(estimated_state[0, 0])
    estimated_velocity_list.append(estimated_state[1, 0])

    # Increment time
    time += dt

# Plotting
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(time_list, true_altitude_list, label='True Altitude (m)')
plt.plot(time_list, measured_altitude_list, label='Measured Altitude (m)', linestyle='dotted')
plt.plot(time_list, estimated_altitude_list, label='Estimated Altitude (m)', linestyle='dashed')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.title('Rocket Ascent with Kalman Filter')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time_list, true_velocity_list, label='True Velocity (m/s)')
plt.plot(time_list, estimated_velocity_list, label='Estimated Velocity (m/s)', linestyle='dashed')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time_list, [alt - est_alt for alt, est_alt in zip(true_altitude_list, estimated_altitude_list)], label='Altitude Estimation Error (m)', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Estimation Error (m)')
plt.legend()

plt.tight_layout()
plt.show()
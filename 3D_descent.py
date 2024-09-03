import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function
def func(x, y):
    return x**2 + y**2

# Numerical gradient calculation using central difference
def numerical_gradient(func, x, y, h=1e-8):
    grad_x = (func(x + h, y) - func(x - h, y)) / (2 * h)
    grad_y = (func(x, y + h) - func(x, y - h)) / (2 * h)
    return np.array([grad_x, grad_y])

# Gradient Descent Algorithm
def gradient_descent(initial_x, initial_y, learning_rate, num_iterations):
    x, y = initial_x, initial_y
    for i in range(num_iterations):
        grad = numerical_gradient(func, x, y)
        x = x - learning_rate * grad[0]
        y = y - learning_rate * grad[1]
        
        # Optionally print the progress
        if i % 100 == 0:
            print(f"Iteration {i}: x = {x:.4f}, y = {y:.4f}, f(x, y) = {func(x, y):.4f}")
    
    return x, y

# Parameters for Gradient Descent
initial_x = 10.0
initial_y = 10.0
learning_rate = 0.1
num_iterations = 2

# Run gradient descent
final_x, final_y = gradient_descent(initial_x, initial_y, learning_rate, num_iterations)
final_z = func(final_x, final_y)

print(f"Minimum found at: x = {final_x:.4f}, y = {final_y:.4f}, f(x, y) = {func(final_x, final_y):.4f}")

# Create a meshgrid for plotting
x = np.linspace(-15, 15, 400)
y = np.linspace(-15, 15, 400)
X, Y = np.meshgrid(x, y)
Z = func(X, Y)

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

# Plot the converged point
ax.scatter(final_x, final_y, final_z, color='red', s=50, label='Minima')

# Labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('f(x, y)')
ax.set_title('Surface plot of f(x, y) = x^2 + y^2')

# Legend
ax.legend()

plt.show()

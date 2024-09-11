import jax
from jax import grad
import jax.numpy as jnp

def func(x):
    
    #y1 = x[0] * x[1] + x[2] * (x[0] + x[1])
    y2 = x[0] * x[1] + x[2] * (x[0] + x[1])
    
    return y2

x = jnp.array([1.0, 2.0, 3.0])
gradient_func = grad(func)
gradients = gradient_func(x)
print(f"Gradients via grad():{gradients}")

gradient_func = jax.jacfwd(func)
gradients = gradient_func(x)
print(f"Gradients via jacfwd:{gradients}")

gradient_func = jax.jacrev(func)
gradients = gradient_func(x)
print(f"Gradients via jacrev:{gradients}")

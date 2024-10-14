import jax
from jax import grad
import jax.numpy as jnp

# ============================================================================#
# def func(x,a):
    
#     y1 = x[0] * x[1] + x[2] * (x[0] + x[1])
#     y2 = x[0] * x[1] + x[2] * (x[0] + x[1])
    
#     return [y1, a*y2]

# x = jnp.array([1.0, 2.0, 3.0])
# # gradient_func = grad(func)
# # gradients = gradient_func(x,1)
# # print(f"Gradients via grad():{gradients}")

# gradient_func = jax.jacfwd(func)
# gradients = gradient_func(x,2)
# print(f"Gradients via jacfwd:{gradients}")

# gradient_func = jax.jacrev(func)
# gradients = gradient_func(x,2)
# print(f"Gradients via jacrev:{gradients}")

# ============================================================================#
import numpy as np 

def func(x):
    
    # y1 = 1/x[0]**2
    # y2 = 1/x[1]**2
    
    # F = jnp.array([[y1, 0], [0, y2]])
    # G = jnp.linalg.inv(F)
    # G = np.reshape(G,(4,))
    G = jnp.zeros((2,2))
    
    G = G.at[0].set(x[0])
    G = G.at[1].set(x[0])
    
    return G

x = jnp.array([1.0, 2.0])

gradient = jax.jacfwd(func, argnums=0)(x)
print(gradient)

# ============================================================================#
# def func(xdict):
    
#     x = xdict["vars"]

    
#     y1 = 1/x[0]**2
#     y2 = 1/x[1]**2
    
#     F = jnp.array([[y1, 0], [0, y2]])
#     G = jnp.linalg.inv(F)
    
#     res_dict = {"G":G.flatten(), "xy":x[0]*x[1]}
    
#     return res_dict

# x = {"vars":jnp.array([1.0, 2.0]), "filler":jnp.array([1.0, 2.0])}
# # x = jnp.array([1.0, 2.0])

# gradient = jax.jacfwd(func, argnums=0)(x)
# print(gradient["G"]["vars"])
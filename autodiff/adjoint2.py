import numpy as np
import jax
import jax.numpy as jnp

def residual(u, x, parameters):
    
    u0, dt  = parameters
    
    res = jnp.zeros(u.shape)
    uprev = jnp.array([[u0]])
    F = jnp.log(x)
    
    for i in range(len(u)):
        val = u[i] - uprev[0] - F*dt
        res = res.at[i].set(val[0])
        uprev.at[0,0].set(u[i])
        
    return res

    
def solve(x, parameters):
    
    global T
    
    u0, dt  = parameters
    u = np.zeros([int(T/dt),1])
    
    uprev = np.array([[u0]])
    F = np.log(x)
    
    
    for i in range(int(T/dt)):
        u[i][0] = uprev[0][0] + F*dt
        uprev[0][0] = u[i][0]
        
    return u

def objective(u,x):
    sum = 0.0
    for i in range(len(u)):
        sum += u[i]
        
    return x*sum


u = jnp.ones([10,])
x = 10.0
parameters = [10.0, 1.0] # u0, dt
T = 10

dr_du = jax.jacrev(residual, argnums=0)(u, x, parameters)
dr_dx = jax.jacrev(residual, argnums=1)(u, x, parameters)

phi = np.linalg.solve(dr_du, dr_dx)

df_du = jax.jacrev(objective, argnums=0)(u,x)
df_dx = jax.jacrev(objective, argnums=1 )(u,x)

Df_Dx = df_dx - df_du @ phi

print("Direct method:", Df_Dx)

"======== Adjoint calculations ========="
psi = np.linalg.solve(np.array(dr_du).T, df_du)
Df_Dx = df_dx - psi.T @ dr_dx

print("Adjoint method:", Df_Dx)


    
# u = jnp.ones([10,1])
# x = 10.0
# parameters = [10.0, 1.0] # u0, dt
# T = 10

# res = residual(u, x, parameters)
# print(res)

# u = solve(x, parameters)

# res = residual(u, x, parameters)
# print(res)
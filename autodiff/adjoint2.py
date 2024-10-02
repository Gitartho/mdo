import numpy as np
import jax
import jax.numpy as jnp

def residual(u, x, parameters):
    
    u0, dt  = parameters
    
    res = jnp.zeros(u.shape)
    uprev = jnp.append(u0, u)
    F = jnp.log(x)
    
    for i in range(len(u)):
        val = u[i] - uprev[i] * (1 + F*dt)
        res = res.at[i].set(val)
        
    return res

    
def solve(x, parameters):
    
    global T
    
    u0, dt  = parameters
    u = np.zeros([int(T/dt),])
    
    uprev = np.array([u0])
    F = np.log(x)
    
    
    for i in range(int(T/dt)):
        u[i] = uprev[0] * (1 + F*dt)
        uprev[0] = u[i]
        
    return u

def objective(u,x):
    sum = 0.0
    for i in range(len(u)):
        sum += u[i]
        
    return x*sum


# u = jnp.ones([10,])
x = 10.0
parameters = [10.0, 0.1] # u0, dt
T = 1
u = solve(x, parameters)

dr_du = jax.jacrev(residual, argnums=0)(u, x, parameters)
dr_dx = jax.jacrev(residual, argnums=1)(u, x, parameters)

print(dr_du)

phi = np.linalg.solve(dr_du, dr_dx)

df_du = jax.jacrev(objective, argnums=0)(u,x)
df_dx = jax.jacrev(objective, argnums=1 )(u,x)

Df_Dx = df_dx - df_du @ phi

print("Direct method:", Df_Dx)

"=========== Adjoint calculations ==========="
psi = np.linalg.solve(np.array(dr_du).T, df_du)
Df_Dx = df_dx - psi.T @ dr_dx

print("Adjoint method:", Df_Dx)

"============= Finite Difference ============"

dx = 0.01
u_pdx = solve(x+dx, parameters)
u_mdx = solve(x-dx, parameters)

obj_pdx = objective(u_pdx, x+dx)
obj_mdx = objective(u_mdx, x-dx)

Df_dx = (obj_pdx - obj_mdx)/(2*dx)
print("Finite Difference", Df_Dx)

"==========================================="
"plotting the analytic vs adjoint derivative"

    
# for x in range(1,10,100):
    
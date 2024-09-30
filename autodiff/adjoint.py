import numpy as np
import jax
import jax.numpy as jnp

# Governing equations
def r(u,x):

    eqn1 = 0.25 * u[0]**2 + u[1]**2 - x
    eqn2 = 4 * u[0] * u[1] - 2*jnp.pi*x 
    return jnp.array([eqn1, eqn2])

# Objective Function
def f(u,x):

    val = u[0]*u[1] + x*u[0] + x*u[1]
    return val


def solve(u0, x, tol=1e-6, max_iter=100):

    u = np.array(u0, dtype=float)

    for i in range(max_iter):

        res = r(u, x)

        if np.linalg.norm(res, ord=2) < tol:
            print(f"Converged after {i} iterations.")
            return u
        
        Jac = jax.jacfwd(r)
        J = Jac(u, x)

        du = jnp.linalg.solve(J, -res)
        u = u + du

    raise ValueError(
        "Newton's method did not converge after the maximum number of iterations.")


"Direct computations"
# Compute Df/Du at u = [1, 2], x = 3

u = [1.0, 1.0]
x = 1.0

dr_du = jax.jacrev(r, argnums=0)(u,x)
dr_dx = jax.jacrev(r, argnums=1)(u,x)

phi = np.linalg.solve(dr_du, dr_dx)

df_du = jax.jacrev(f, argnums=0)(u,x)
df_dx = jax.jacrev(f, argnums=1 )(u,x)

Df_Dx = df_dx - df_du @ phi

print("Direct method:", Df_Dx)

"Adjoint calculations"
psi = np.linalg.solve(np.array(dr_du).T, df_du)
Df_Dx = df_dx - psi.T @ dr_dx

print("Adjoint method:", Df_Dx)
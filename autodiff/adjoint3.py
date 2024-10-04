"Adjoint method code using classes"

import jax
from jax import numpy as np

class system():
    
    def __init__(self, F, Nx, Nt, param_dict):
        
        self.F = F #transition matrices
        self.Nx = Nx
        self.Nt = Nt
        self.u = np.zeros([Nx, Nt])    
        
        self.x0 = param_dict["Xinit"]
        self.dt = param_dict["timestep"]
        
    def residual(self, x, u):
        
        Nx = self.Nx
        Nt = self.Nt
        x0 = self.x0
        dt = self.dt
        F = self.F
        
        res = np.array([Nt*Nx, 1])
        xprev = np.append(x0, x)
        
        for i in range(Nt):
            
            j = 4*i
            # F = np.array([[u[j]  , u[j+1]], 
            #               [u[j+2], u[j+3]]])
            
            val = x[:,i] - xprev[:,i] - dt * F @ xprev[:,i]
            res = res.at[Nx*i : Nx*(i+1)].set(val)
            
        return res
    
    def solve(self, u, tol=1e-6, max_iter=100):
        
        Nx = self.Nx
        Nt = self.Nt
        x0 = self.x0
        dt = self.dt
        
        # Solving r(x, u) is a root finding operation.
        # Note the dimensions; x -> Nx X Nt; residual -> Nx*Nt X 1
        
        x = np.ones([Nx, Nt])
        
        for i in range(max_iter):
            
            res = self.residual(x, u)

            if np.linalg.norm(res, ord=2) < tol:
                print(f"Converged after {i} iterations.")
                return x
            
            Jac = jax.jacfwd(self.residual, argnums=0)
            dr_dx = Jac(x,u)

            dx = jnp.linalg.solve(dr_dx, -res)
            x = x + dx

        raise ValueError(
            "Newton's method did not converge after the maximum number of iterations.")

F = np.arrray([[2,0],[0,2]])
x0 = np.arrray([[1],[1]])
dt = 1

param = {}

    
system()
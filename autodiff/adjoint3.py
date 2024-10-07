"Adjoint method code using classes"

import jax
from jax import numpy as np

class system():
    
    #=========================================================================#
    def __init__(self, F, Nx, Nt, param_dict):
        
        self.F = F #transition matrices
        self.Nx = Nx
        self.Nt = Nt
        self.x = np.zeros([Nx, Nt])
        
        self.x0 = param_dict["Xinit"]
        self.dt = param_dict["timestep"]
    
    #=========================================================================#
    def residual(self, x, u):
        
        Nx = self.Nx
        Nt = self.Nt
        x0 = self.x0
        dt = self.dt
        F = self.F
        
        x = np.reshape(x, (Nx, Nt), order='F')
        
        res = np.ones([Nt*Nx,])
        xprev = np.append(x0, x, axis=1)
        
        for i in range(Nt):
            
            j = 4*i
            F = np.array([[u[j]  , u[j+1]], 
                          [u[j+2], u[j+3]]])
            
            val = x[:,i] - xprev[:,i] - dt * F @ xprev[:,i]
            res = res.at[Nx*i : Nx*(i+1)].set(val)
        
        return res
    
    #=========================================================================#
    def solve(self, u, tol=1e-6, max_iter=100):
        
        Nx = self.Nx
        Nt = self.Nt
        x0 = self.x0
        dt = self.dt
        
        # Solving r(x, u) is a root finding operation.
        # Note the dimensions; x -> Nx X Nt; residual -> Nx*Nt X 1
        
        x = np.ones([Nx*Nt,])
        
        for i in range(max_iter):
            
            res = self.residual(x, u)

            if np.linalg.norm(res, ord=2) < tol:
                print(f"Converged after {i} iterations.")
                x_sol = np.reshape(x, (Nx, Nt), order='F')
                self.x = x_sol
                return x_sol
            
            Jac = jax.jacfwd(self.residual, argnums=0)
            dr_dx = Jac(x,u)
            
            print(dr_dx.shape)

            dx = np.linalg.solve(dr_dx, -res)
            x = x + dx

        raise ValueError(
            "Newton's method did not converge after the maximum number of iterations.")
    
    #=========================================================================#
    def objective(self, x, u):
        
        Nx = self.Nx
        Nt = self.Nt
        
        obj = np.linalg.norm(x, ord=1)
        
        return obj
    
#=============================================================================#
" Compute d(objctive)/d(control u) "

def adjoint(sys_obj, u):
    
    x = sys_obj.x.flatten()
    
    dr_dx = jax.jacrev(sys_obj.residual, argnums=0)(x,u)
    dr_du = jax.jacrev(sys_obj.residual, argnums=1)(x,u)
    
    print(f"dr_dx {dr_dx.shape}; dr_du {dr_du.shape}")
    
    phi = np.linalg.solve(dr_dx, dr_du)
    
    df_dx = jax.jacrev(sys_obj.objective, argnums=0)(x,u)
    df_du = jax.jacrev(sys_obj.objective, argnums=1)(x,u)
    
    Df_Du = df_du - df_dx @ phi

    print("Direct method:", Df_Du)

    "======== Adjoint calculations ========="
    psi = np.linalg.solve(np.array(dr_dx).T, df_dx)
    Df_Du = df_du - psi.T @ dr_du

    print("Adjoint method:", Df_Du)
    
#=============================================================================#

F = np.array([[2,0],[0,2]])
x0 = np.array([[1.0],[1.0]])
dt = 1
param = {"Xinit":x0, "timestep":dt}
u = np.array([10.0])
    
sys_obj = system(F, 2, 10, param)

adjoint(sys_obj, u)


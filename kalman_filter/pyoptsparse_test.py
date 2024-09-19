"""
Created on Thu Sep 19 14:05:48 2024

@author: gitartha
"""

from pyoptsparse import OPT, Optimization

def obfunc(xdict):
    
    x = xdict["list1"]
    
    func = {}
    func["cost"] = x[0]**2 + x[1]**2
    
    con = [0]*2
    con[0] = x[0]
    con[1] = x[1]
    
    func["constraint"] = con
    
    fail = False
    
    return func, fail

problem = Optimization("test problem", obfunc)

ubound = [5.0, 5.0]
lbound = [-5.0, -5.0]
guess = [-1.0, -1.0]

problem.addVarGroup("list1", 2, "c", value=guess, lower=lbound, upper=ubound)
problem.addConGroup("constraint", 2, lower = None, upper=None)
problem.addObj("cost")

#print(problem)

optimizer = OPT("ipopt")
solution = optimizer(problem, sens="FD")
    
print(solution)

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 19:13:15 2024

@author: HP User
"""

class sample:
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2
        self.func = lambda: None
        
    def func1(self):
        return self.var1 + self.var2
    
    def change(self):        
        self.func = self.func1
        
    def variable(self, other):
        variable = sample(self.var1, other.var2)
        def func1():
            return self.func() + other.func()
        variable.func = func1
        return variable
    

a = sample(2, 3)
a.change()

s = a.variable(a)

print(s.func())

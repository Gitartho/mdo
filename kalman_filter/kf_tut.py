#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:39:13 2024

@author: gitartha

A TUTORIAL ON KALMAN FILTER
"""

class system:
    
    state = [] 
    var = []
    
    def __init__(self, state, var):
        
        for s,v in zip(state,var):
            self.state.append(s)
            self.var.append(v)
    
    def update(self, prev, z)
        
        n = 10


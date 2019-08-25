# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:23:51 2019

@author: hweem
"""
import numpy as np

class Generate_Samples:
    def __init__(self):
        pass
    
    def gen_samples(self, x_min, x_max, mx):
        #X.shape = (nx, mx)
        X = np.linspace(x_min, x_max, num = mx, endpoint = False).reshape(1, mx)
        y_lin = .1 * X + np.cos(np.pi * X) / X
        y_noise = np.random.randn(1, mx) * .05
        
        #Y.shape = (1, mx)
        Y = y_lin + y_noise
        
        return X, Y
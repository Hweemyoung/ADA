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
    
    def split_samples(self, X, Y, num_set, p):
        #X.shape = (nx, mx)
        #Y.shape = (1, mx)
        size = X.shape[1] // num_set
        #shuffle
        X = X[:, p]
        Y = Y[:, p]
        train_x_list = [np.concatenate((X[:, : i * size], X[:, (i + 1) * size :]), axis = 1) for i in range(num_set)]
        train_y_list = [np.concatenate((Y[:, : i * size], Y[:, (i + 1) * size :]), axis = 1) for i in range(num_set)]
        test_x_list = [X[:, i * size : (i + 1) * size] for i in range(num_set)]
        test_y_list = [Y[:, i * size : (i + 1) * size] for i in range(num_set)]
        
        return train_x_list, train_y_list, test_x_list, test_y_list
    

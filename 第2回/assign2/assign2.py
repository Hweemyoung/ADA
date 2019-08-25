# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 00:36:26 2019

@author: hweem
"""

import numpy as np

class Gauss_Kernel:
    def __init__(self, l, h, p):
        self.l = l
        self.h = h
        self.X = None #.shape = (nx, mx)
        self.Y = None #.shape = (1, mx)
        self.p = p #shuffle
        self.parameters = None #.shape = (1, mx)
        self.pred_y = []
        self.loss = []
        
    def gen_samples(self, x_min, x_max, m):
        self.X = np.linspace(x_min, x_max, num = m, endpoint = False).reshape(1, m)
        y_lin = .1 * self.X + np.cos(np.pi * self.X) / self.X
        y_noise = np.random.randn(1, m) * .05
        self.Y = y_lin + y_noise
    
    def split_samples(self, X, Y, num_set):
        #X.shape = (nx, mx)
        #Y.shape = (1, mx)
        size = X.shape[1] // num_set
        #shuffle
        X = X[:, self.p]
        Y = Y[:, self.p]
        self.train_x_list = [np.concatenate((X[:, : i * size], X[:, (i + 1) * size :]), axis = 1) for i in range(num_set)]
        self.train_y_list = [np.concatenate((Y[:, : i * size], Y[:, (i + 1) * size :]), axis = 1) for i in range(num_set)]
        self.test_x_list = [X[:, i * size : (i + 1) * size] for i in range(num_set)]
        self.test_y_list = [Y[:, i * size : (i + 1) * size] for i in range(num_set)]
    
    def optimize_kernel(self, X, Y, l, h):
        #X.shape = (nx, mx)
        #Y.shape = (1, mx)
        K = self.gen_mat_kernel(X, X, h) #.shape = (mx, mx)
        
        #.shape = (1, mx)
        self.parameters = np.linalg.inv(K.T.dot(K) + l * np.identity(K.shape[0])).dot(K.T).dot(Y.T).T
    
    def gen_mat_kernel(self, X, C, h):
        #X.shape = (nx, mx)
        #C.shape = (nx, mc)
        mat_kernel = X[:, :, None] - C[:, None] #.shape = (nx, mx, mc)
        
        #.shape = (mx, mc)
        return np.exp(- np.linalg.norm(mat_kernel, axis = 0, keepdims = False) / 2 / h**2 )

    def cross_val(self, train_x, test_x, test_y, h, parameters):
        K_test = self.gen_mat_kernel(train_x, test_x, h) #.shape = (mtrain, mtest)
        pred_y = K_test.T.dot(parameters.T).T #.shape = (1, mtest)
        self.pred_y.append(pred_y)
        self.loss.append(np.linalg.norm(pred_y - test_y))
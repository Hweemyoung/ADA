# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 00:36:26 2019

@author: hweem
"""

import numpy as np

class Robust:
    def __init__(self, mx, e):
        #self.l = l
        #self.h = h
        #self.p = p #shuffle
        self.X = None #.shape = (nx, mx)
        self.Y = None #.shape = (1, mx)
        self.PI = None
        self.mx = mx
        self.parameters = {}
        self.e = e
        self.pred_y = []
        self.loss = []
        self.epochs = []
        self.train_x_list, self.train_y_list, self.test_x_list, self.test_y_list \
            = None, None, None, None
        
    def initialize_parameters(self, mx):
        self.parameters['theta'] = np.random.randn(1, mx)
        self.parameters['z'] = np.random.randn(1, mx)
        self.parameters['u'] = np.random.randn(1, mx)

    def optimize_robust(self, PI, parameters, X, Y, threshold = .3):
        # r.shape = (mx, mx)
        # e: constant
        parameters = parameters
        num_epoch = 0
        while True:
            theta = parameters['theta']
            r = parameters['r']
            w = (np.abs(r) <= self.e) + (np.abs(r) > self.e) * np.divide(self.e, r)
            W = np.diag(w)
            new_theta = np.linalg.inv(PI.T.dot(W).dot(PI)).dot(PI.T).dot(W).dot(Y)
            f_temp = self.gen_f(theta)
            new_r = f_temp(X) - Y #.shape = (1, m)
            parameters['theta'] = new_theta
            parameters['r'] = new_r

        self.epochs.append(num_epoch)
        
    def gen_f(self, theta):
        return 
    
    def gen_mat_kernel(self, X, C, h):
        #X.shape = (nx, mx)
        #C.shape = (nx, mc)
        mat_kernel = X[:, :, None] - C[:, None] #.shape = (nx, mx, mc)
        
        #.shape = (mx, mc)
        return np.exp(- np.linalg.norm(mat_kernel, axis = 0, keepdims = False) / 2 / h**2 )
    
    def gen_PI(self, X, ground_functions):
        # X.shape = (nx, mx)
        # ground_functions return arr(.shape = (n,1))
        ground_functions()

    def cross_val(self, train_x, test_x, test_y, h, parameters):
        K_test = self.gen_mat_kernel(train_x, test_x, h) #.shape = (mtrain, mtest)
        theta = self.parameters['theta']
        pred_y = K_test.T.dot(theta.T).T #.shape = (1, mtest)
        self.pred_y.append(pred_y)
        self.loss.append(np.linalg.norm(pred_y - test_y))
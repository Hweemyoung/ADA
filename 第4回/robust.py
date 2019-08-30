# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 00:36:26 2019

@author: hweem
"""

import numpy as np

class Robust:
    def __init__(self, train_x, train_y, target_theta, fs, e):
        #self.l = l
        #self.h = h
        #self.p = p #shuffle
        
        self.train_x = train_x
        self.train_y = train_y
        self.target_theta = target_theta
        self.fs = fs
        
        self.mtrain = train_x.shape[1]
        self.nx = train_x.shape[0]
        self.ny = train_y.shape[0]
        self.n = len(fs)
        
        self.PI = self.gen_PI()
        self.parameters = {}
        self.e = e
        self.pred_y = []
        self.loss = []
        self.epochs = None
        
    def initialize_parameters(self, n):
        self.parameters['theta'] = np.random.randn(1, n)
        self.parameters['r'] = np.random.randn(1, n)
        self.parameters['u'] = np.random.randn(1, n)

    def target_function(self, X, theta, ny):
        Y = np.zeros((ny, X.shape[1]))
        fs = self.gen_fs()
        for j in len(fs):
            Y += theta[j] * fs['f' + str(j)](X)
        return Y

    def optimize_robust(self, PI, parameters, X, threshold = .3):
        # r.shape = (mx, mx)
        # e: constant
        parameters = parameters
        num_epoch = 0
        while True:
            num_epoch += 1
            theta = parameters['theta']
            Y_hat = self.target_function(X, theta, self.ny)
            #build W
            r = np.abs(Y_hat - self.train_y) # residue
            w = (np.abs(r) <= self.e) + (np.abs(r) > self.e) * np.divide(self.e, r)
            W = np.diag(w)
            #update theta
            new_theta = np.linalg.inv(PI.T.dot(W).dot(PI)).dot(PI.T).dot(W).dot(self.train_y)
            parameters['theta'] = new_theta
            l2 = np.linalg.norm(new_theta - self.target_theta)
            if num_epoch % 100 == 0:
                print('epoch: ' + str(num_epoch) + '\t l2 = ' + str(l2))
            if l2 < threshold:
                break
        print('epoch: ' + str(num_epoch) + '\t l2 = ' + str(l2))
        self.epochs = num_epoch
    
    def gen_PI(self, fs, X):
        PI = 
        
        return 
    
    def gen_mat_kernel(self, X, C, h):
        #X.shape = (nx, mx)
        #C.shape = (nx, mc)
        mat_kernel = X[:, :, None] - C[:, None] #.shape = (nx, mx, mc)
        
        #.shape = (mx, mc)
        return np.exp(- np.linalg.norm(mat_kernel, axis = 0, keepdims = False) / 2 / h**2 )


    def cross_val(self, train_x, test_x, test_y, h, parameters):
        K_test = self.gen_mat_kernel(train_x, test_x, h) #.shape = (mtrain, mtest)
        theta = self.parameters['theta']
        pred_y = K_test.T.dot(theta.T).T #.shape = (1, mtest)
        self.pred_y.append(pred_y)
        self.loss.append(np.linalg.norm(pred_y - test_y))
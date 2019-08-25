# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 00:36:26 2019

@author: hweem
"""

import numpy as np

class Sparse_Regression:
    def __init__(self, l, h, p, mx):
        self.l = l
        self.h = h
        self.p = p #shuffle
        self.X = None #.shape = (nx, mx)
        self.Y = None #.shape = (1, mx)
        self.K = None
        self.mx = mx
        self.parameters = {}
        self.pred_y = []
        self.loss = []
        self.epochs = []
        self.train_x_list, self.train_y_list, self.test_x_list, self.test_y_list \
            = None, None, None, None
        
    def initialize_parameters(self, mx):
        self.parameters['theta'] = np.random.randn(1, mx)
        self.parameters['z'] = np.random.randn(1, mx)
        self.parameters['u'] = np.random.randn(1, mx)

    def optimize_sparse(self, PI, parameters, l, Y, threshold = .3):
        num_epoch = 0
        temp1 = np.linalg.inv(PI.T.dot(PI) + np.identity(self.mx))
        temp2 = PI.T.dot(Y.T)
        while True:
            num_epoch += 1
            print('epoch: ' + str(num_epoch))
            theta = self.parameters['theta']
            z = self.parameters['z'] #.shape = (1, mx)
            u = self.parameters['u'] #.shape = (1, mx)
            #.shape = (1, mx)
            new_theta = temp1.dot(temp2 + z.T - u.T).T            #.shape = (1, mx)
            new_z = np.max(np.concatenate([np.zeros((1, self.mx)), new_theta + u - l * np.ones((1, self.mx))]), axis = 0, keepdims = True) \
                - np.max(np.concatenate([np.zeros((1, self.mx)), - new_theta - u - l * np.ones((1, self.mx))]), axis = 0, keepdims = True)
            #.shape = (1, mx)
            new_u = u + new_theta - new_z
            self.parameters['theta'] = new_theta
            self.parameters['z'] = new_z
            self.parameters['u'] = new_u
            l2_t = np.linalg.norm(new_theta - theta)
            l2_z = np.linalg.norm(new_z - z)
            l2_u = np.linalg.norm(new_u - u)
            print('l2_t = ' + str(round(l2_t, 5)))
            print('l2_z = ' + str(round(l2_z, 5)))
            print('l2_u = ' + str(round(l2_u, 5)))
            if l2_t < threshold:
                break
        self.epochs.append(num_epoch)
    
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
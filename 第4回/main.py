# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 22:49:35 2019

@author: hweem
"""
import numpy as np
import robust, gen_samples

if __name__ == '__main__':
    #10^-1 ~ 10^1
    #r = np.random.rand(5)
    #hs = 10 ** (r * 2 - 1)
    #ls = 10 ** (r * 2 - 1)
    
    sampler = gen_samples.Generate_Samples()
    num_samples = 1001
    num_set = 5
    X, Y, target_theta = sampler.gen_samples(-5, 5, num_samples) #X.shape = (1, mx), Y.shape = (1, mx)
    p = np.random.permutation(num_samples) #shuffle
    X, Y = X[:,p], Y[:,p]
    train_x_list, train_y_list, test_x_list, test_y_list \
        = sampler.split_samples(X, Y, num_set, p)
    
    fs = sampler.gen_fs()
    model_dict = {}
    loss_hp = {}
    
    e = .1
    n = 2
    
    for (train_x, train_y, test_x, test_y) \
        in zip(train_x_list, train_y_list, test_x_list, test_y_list):
        model = robust.Robust(train_x, train_y, target_theta, fs, e)
        model.initialize_parameters(model.n)
        #define PI(=K) for training set
        #iterate optimization
        #cross-validate
        model.cross_val(train_x, test_x, test_y, model.h, model.parameters)
        model_dict[(model.h, model.l)] = model
    loss_hp[(model.h, model.l)] = sum(model.loss) / len(model.loss)
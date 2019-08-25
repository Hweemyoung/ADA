# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 22:49:35 2019

@author: hweem
"""
import numpy as np
import assign2

if __name__ == '__main__':
    #10^-1 ~ 10^1
    r = np.random.rand(5)
    hs = 10 ** (r * 2 - 1)
    ls = 10 ** (r * 2 - 1)
    num_samples = 1001
    p = np.random.permutation(num_samples) #shuffle
    model_dict = {}
    loss_hp = {}
    num_set = 5
    for h in hs:
        for l in ls:
            model = assign2.Gauss_Kernel(l, h, p)
            model.gen_samples(-5, 5, num_samples)
            model.split_samples(model.X, model.Y, num_set)
            for (train_x, train_y, test_x, test_y) \
                in zip(model.train_x_list, model.train_y_list, model.test_x_list, model.test_y_list):                
                
                model.optimize_kernel(train_x, train_y, model.l, model.h)
                model.cross_val(train_x, test_x, test_y, model.h, model.parameters)
            model_dict[(model.h, model.l)] = model
            loss_hp[(model.h, model.l)] = sum(model.loss) / len(model.loss)
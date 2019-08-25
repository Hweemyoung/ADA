#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)  # set the random seed for reproducibility


# In[133]:


# サンプル生成
def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    pix = np.pi * x
    target = np.sin(pix) / pix + 0.1 * x
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
    return x, target + noise

sample_size = 100
xmin, xmax = -3, 3
x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)

vis_x = np.linspace(start=xmin, stop=xmax, num=1000) # 識別面可視化用サンプル


# In[134]:


# 交差検証サンプル生成
num_split = 5
num_sample_test = int(sample_size / num_split)
num_sample_train = int(sample_size - num_sample_test)

x, y = zip(*np.random.permutation(list(zip(x, y))))
x, y = np.array(x), np.array(y)

test_x = x[0:num_sample_test]
test_y = y[0:num_sample_test]

train_x = x[num_sample_test:]
train_y = y[num_sample_test:]


# In[136]:


# hyper parameter
h = 0.3
l = 0.01


# In[137]:


def calc_design_matrix(x, c, h):
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))


# In[150]:


# calculate design matrix
k_train = calc_design_matrix(train_x, train_x, h)

# initialize
z = np.random.randn(num_sample_train)
u = np.random.randn(num_sample_train)

iter_num = 50

# update parameters
for i in range(iter_num):
    theta = np.linalg.inv(k_train.T.dot(k_train) +np.identity(len(k_train))).dot(k_train.T.dot(train_y) + z - u)    
    z = np.maximum(0, theta + u - l*np.ones(num_sample_train)) + np.minimum(0, theta + u + l*np.ones(num_sample_train))
    u = u + theta - z


# In[152]:


# calculate design matrix for test
k_test = calc_design_matrix(train_x, test_x, h)
pred_y = k_test.dot(theta)

# create data to visualize the prediction
vis_k = calc_design_matrix(train_x, vis_x, h)
vis_y = vis_k.dot(theta)


# In[155]:


fig = plt.figure(figsize=(12, 8))
plt.scatter(train_x, train_y, c='green', marker='o')
plt.plot(vis_x, vis_y)
plt.savefig('assign3_2_sparse_regression.png')


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


# In[2]:

# In[3]:


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


# In[4]:


# 交差検証サンプル生成

num_split = 5
num_sample = int(sample_size / num_split)

train_x_list = [np.concatenate([x[0:i*num_sample], x[(i+1)*num_sample:]]) for i in range(num_split)]
train_y_list = [np.concatenate([y[0:i*num_sample], y[(i+1)*num_sample:]]) for i in range(num_split)]
test_x_list = [x[i*num_sample:(i+1)*num_sample] for i in range(num_split)]
test_y_list = [y[i*num_sample:(i+1)*num_sample] for i in range(num_split)]

assert all([len(train_x) == sample_size - num_sample for train_x in train_x_list])
assert all([len(train_y) == sample_size - num_sample for train_y in train_y_list])

assert all([len(test_x) == num_sample for test_x in test_x_list])
assert all([len(test_y) == num_sample for test_y in test_y_list])


# In[5]:


# ハイパーパラメータ
hyp_loss = {}
hs = [0.03, 0.3, 3]
ls = [0.001, 0.01, 0.1]


# In[6]:


# 交差検証
def calc_design_matrix(x, c, h):
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))

fig = plt.figure(figsize=(18, 12))
subplot_idx = 100*(len(hs)) + 10*(len(ls))

for h in hs:
    for l in ls:
        subplot_idx += 1
        ax = fig.add_subplot(subplot_idx)
        losses = []
        
        for train_x, train_y, test_x, test_y in zip(train_x_list, train_y_list, test_x_list, test_y_list):
            # calculate design matrix
            k_train = calc_design_matrix(train_x, train_x, h)

            # solve the least square problem
            theta = np.linalg.solve(k_train.T.dot(k_train) + l * np.identity(len(k_train)), k_train.T.dot(train_y[:, None]))
            
            # calculate design matrix
            k_test = calc_design_matrix(train_x, test_x, h)
            pred_y = k_test.dot(theta)[:, 0]
            
            # calculate loss
            loss = np.linalg.norm(test_y - pred_y)
            losses += [loss]
        
        # save 
        loss_mean = np.mean(losses)
        hyp_loss[(h, l)] = loss_mean
        
        # create data to visualize the prediction
        vis_k = calc_design_matrix(train_x, vis_x, h)
        vis_y = vis_k.dot(theta)

        title = 'h: ', h, 'l: ', l, 'loss:', '%.3f' % loss_mean
        ax.set_title(title)
        ax.scatter(train_x, train_y, c='green', marker='o')
        ax.plot(vis_x, vis_y)
        
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
print('ガウス幅: %.3f, 正則化係数: %.3f' % sorted(hyp_loss.items(), key=lambda x:x[1])[0][0])


# In[ ]:





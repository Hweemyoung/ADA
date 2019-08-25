#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
np.random.seed(1)

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[2]:


# データ生成
def data_generate(n=50):
    x = np.random.randn(n, 3)
    x[:n // 2, 0] -= 15
    x[n // 2:, 0] -= 5
    x[1:3, 0] += 10
    x[:, 2] = 1
    y = np.concatenate((np.ones(n // 2), -np.ones(n // 2)))
    index = np.random.permutation(np.arange(n))
    return x[index], y[index]

X, Y = data_generate()


# In[3]:


# 各クラスのサンプル、サンプル数
n = len(X)
cs = np.unique(Y)
indices_cs = [np.where(Y==c) for c in cs]

b = X.shape[1]


# In[4]:


# ハイパーパラメータ
gamma = 0.1
n_epochs = 10


# In[5]:


# 初期化
m = np.random.randn(b)
s = np.random.randn(b, b)


# In[8]:


# 最適化
for epoch in range(n_epochs):
    for x, y in zip(X, Y):
        beta = x.dot(s).dot(x) + gamma
        s = s - s.dot(x[:, None] * x[None, :]).dot(s)/beta
        m = m - (m.dot(x) - y)*s.dot(x)/beta


# In[9]:


# 可視化
x_vis = np.linspace(start=-11, stop=-9, num=1000) 
y_vis = (m[0]*x_vis + m[2])/m[1]

plt.xlim(-20, 0)
plt.ylim(-2, 2)
for indices_c in indices_cs:
    plt.scatter(X[indices_c, 0], X[indices_c, 1])
plt.plot(x_vis, y_vis)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:33:22 2020

eager模式介绍

@author: tinghai
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0


#%% Eager模式（命令行式编写环境/tensroflow的交互模式）
print(tf.executing_eagerly()) #True

# tensor可以与numpy直接运算
a = tf.constant([[1,2],[3,4]])
b = tf.add(a, 1)
c = tf.multiply(a, b)

num = tf.convert_to_tensor(10)
for i in range(num.numpy()):
    i = tf.constant(i)
    if int(i % 2) == 0:
        print(i)

d = np.array([[5,6],[7,8]])
print(a + d)
print((a + d).numpy())

# 变量
v = tf.Variable(0.0)
print(v+1)

v.assign(5) #改变变量的值
v.assign_add(1) #变量值加1
v.read_value() #返回变量值

# 梯度运算
w = tf.Variable([[3.0]]) #需要是float数据类型
with tf.GradientTape() as t:
    loss = w*w + w
grad = t.gradient(loss, w) # 求解loss对w的微分

w = tf.constant([[3.0]])
with tf.GradientTape() as t:
    t.watch(w) # 针对常量w进行跟踪，以便于后续使用t.gradient()求导，Variable不需要watch()
    loss = w*w + w
grad2 = t.gradient(loss, w)

w = tf.constant([[3.0]])
with tf.GradientTape(persistent=True) as t: #persistent=True用于多次计算微分
    t.watch(w)
    y = w*w + w
    z = y*y
grad3 = t.gradient(y, w)
grad4 = t.gradient(z, w)

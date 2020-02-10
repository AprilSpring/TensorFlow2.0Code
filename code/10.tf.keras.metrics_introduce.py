#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:35:41 2020

tf.keras.metrics评价指标

@author: tinghai
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0


#%% 评价指标汇总：tf.keras.metrics()
m = tf.keras.metrics.Mean('acc') #返回计算acc的对象
print(m(10))
print(m(20))
print(m([30,40]))
print(m.result().numpy()) #会保留之前的状态一起计算，返回均值25
m.reset_states() #重制状态

a = tf.keras.metrics.SparseCategoricalAccuracy('acc')
a(labels, predictions) # 自动选择概率最大位置，并计算正确率


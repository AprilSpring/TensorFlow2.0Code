#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:23:22 2020

tf.keras二分类

@author: tinghai
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0

#%% 二分类：logistic model
train_x = np.random.rand(100,3)
train_y = np.random.randint(0,2,(100,1))
test_x = np.random.rand(100,3)
test_y = np.random.randint(0,2,(100,1))
model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'), #第一层需要定义输入数据的维度：input_shape
                             tf.keras.layers.Dense(5, activation='relu'),
                             tf.keras.layers.Dense(1, activation='sigmoid')])
model.summary()
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['acc'])
#history = model.fit(train_x, train_y, epochs=300)
history = model.fit(train_x, train_y, epochs=300, validation_data=(test_x, test_y)) # 在每个epoch上评估测试集准确率
model.predict(test_x)

print(history.history.keys()) # losss, acc, val_loss, val_acc
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()

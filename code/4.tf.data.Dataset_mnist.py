#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:27:58 2020

tf.data.Dataset使用

@author: tinghai
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0

#%% tf.data
# 创建dataset的几种方式
dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5]) 

dataset = tf.data.Dataset.from_tensor_slices([[1,2],[3,4],[5,6]])

dataset = tf.data.Dataset.from_tensor_slices({'a':[1,2,3,4],
                                              'b':[6,7,8,9],
                                              'c':[12,13,14,15]})
dataset = tf.data.Dataset.from_tensor_slices(np.array([1,2,3,4,5])) 

for ele in dataset:
    print(ele)

for ele in dataset:
    print(ele.numpy()) # 转换回numpy数据格式

for ele in dataset.take(4): # 提取topN
    print(ele.numpy())

# shuffle, repeat, batch的使用
dataset = dataset.shuffle(buffer_size=5, seed=0) # 打乱
dataset = dataset.repeat(count=3) # 重复
dataset = dataset.batch(batch_size=3)
for ele in dataset:
    print(ele.numpy())

# 数据变换：map
dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5]) 
dataset = dataset.map(tf.square)
print([ele.numpy() for ele in dataset])

# mnist示例
(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
train_image = train_image/255.0 # 0-1值
test_image = test_image/255.0 # 0-1值
print(train_image.shape) # (60000, 28, 28)

ds_train_img = tf.data.Dataset.from_tensor_slices(train_image)
ds_train_lab = tf.data.Dataset.from_tensor_slices(train_label)
ds_train = tf.data.Dataset.zip((ds_train_img, ds_train_lab)) #两个tensor的对应位置元素合并，((28,28),())

ds_test = tf.data.Dataset.from_tensor_slices((test_image, test_label)) #同ds_train生成的效果一样，((28,28),())

ds_train = ds_train.shuffle(10000).repeat().batch(64)
ds_test = ds_test.batch(64) # 默认使用了repeat()

model = tf.keras.Sequential(tf.keras.layers.Flatten(input_shape=(28, 28)),
                            tf.keras.layers.Dense(128, activation='relu'),
                            tf.keras.layers.Dropout(0.1),
                            tf.keras.layers.Dense(10, activation='softmax'))
model.summary()
model.compile(opitimizer=tf.keras.optimizer.Adam(lr=0.01),
              loss='sparse_categorical_corssentropy', # 当label为顺序数字编码时
              metrics=['acc'])
history = model.fit(ds_train, 
                    epochs=5, 
                    steps_per_epoch=train_image.shape[0]//64, # "//" used to be int
                    validation_data=ds_test, 
                    validation_steps=test_image.shape[0]//64) 


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:36:38 2020

tf.keras中tensorboard使用：mnist数据集

@author: tinghai
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0

#%% tensorboard可视化（keras定义模型）
import os
import datetime

(train_image, train_labels), (test_image, test_labels) = tf.keras.datasets.mnist.load_data()
train_image = tf.expand_dims(train_image, -1) #-1表示扩增的最后一个维度，由于使用CNN因此需要扩增数据维度
train_image = tf.cast(train_image/255, tf.float32) #需要float类型才能做梯度运算
train_labels = tf.cast(train_labels, tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels))
dataset = dataset.shuffle(10000).repeat().batch(32) # 默认repeat(1)；如果使用fit方法的话，需添加repeat(),无限循环

test_image = tf.expand_dims(test_image, -1)
test_image = tf.cast(test_image/255, tf.float32)
test_labels = tf.cast(test_labels, tf.int64)
test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_labels))
test_dataset = test_dataset.batch(32)

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16,[3,3], activation='relu',input_shape=(28,28,1)), #任意图片大小：input_shape=(None,None,1)
        tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
        tf.keras.layser.GlobalMaxPooling2D(), #GlobalAveragePooling2D()
        tf.keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

# tensorboard显示上述模型中定义的评估指标
log_dir = os.path.join('logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# tensorboard显示其他自定义指标
# 以学习速率为例，将LearningRateScheduler()传给model.fit()
file_writer = tf.summary.create_file_writer(log_dir + '/lr') #创建文件编写器
file_writer.set_as_default() #将file_writer设置成默认文件编写器

def lr_sche(epoch):
    learning_rate = 0.2
    if epoch > 5:
        learning_rate = 0.02
    elif epoch > 10:
        learning_rate = 0.01
    else:
        learning_rate = 0.005
    tf.summary.scaler('leaning_rate', data=learning_rate, step=epoch) #收集learning_rate到默认的文件编写器（即file_writer）
    return learning_rate

lr_callback = tf.keras.calllbacks.LearningRateScheduler(lr_sche) #创建lr的回调函数

model.fit(dataset, 
          epochs=10, 
          step_per_epoch=60000//128, 
          validation_data=test_data,
          validation_step=10000/128,
          callbacks=[tensorboard_callback, lr_callback])


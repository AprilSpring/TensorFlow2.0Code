#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:29:02 2020

tf.keras构建CNN

@author: tinghai
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0

#%% CNN
import keras
from keras import layers

layers.Conv2D(filters, #卷积核数量（即卷积后的通道数）
              kernal_size, #卷积核大小
              strides=(1,1), #步长为1
              padding='valid', # 'same'
              activation='relu',
              use_bias=True,
              kernel_initializer='glorot_uniform',
              bias_initializer=None,
              kernel_regularizer=None, #正则化
              bias_regularizer=None)
              
         
layers.MaxPooling2D(pool_size=(2,2),
                    strides=None,
                    padding='valid')

# mnist示例
# !pip install -q tensorflow-gpu==2.0.0-alpha0
tf.test.is_gpu_available()

(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
train_image = np.expand_dims(train_image, -1) # 或reshape(), -1表示扩增的最后一个维度，生成[样本量，长，宽，通道]，与上述使用Flatten不同
test_image = np.expand_dims(test_image, -1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3), 
                                 input_shape=train_image.shape[1:], #shape[1:]表示除去第一维度，即去除batch的维度，首次需要定义该参数
                                 activation='relu',
                                 padding='same'))
print(model.output_shape) #(None, 28, 28 ,32)
model.add(tf.keras.layers.MaxPooling2D()) # default pooling_size=(2,2), #(None, 14, 14 ,32)
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Conv2D(64, (3,3), 
                                 activation='relu',
                                 padding='same')) #(None, 14, 14 ,64)
model.add(tf.keras.layers.MaxPooling2D()) #(None, 7, 7 ,64)
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.GlobalAveragePooling2D()) #全局平均池化，或使用Flatten()使其变成1个维度, (None, 64)
model.add(tf.keras.layers.Dense(128, activation='relu')) #FFN, (None, 128)
model.add(tf.keras.layers.Dense(10, activation='softmax')) #softmax层，(None, 10)
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
history = model.fit(train_image, train_label, epochs=10, validation_data=(test_image, test_label))
print(history.history.keys()) # losss, acc, val_loss, val_acc
plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()

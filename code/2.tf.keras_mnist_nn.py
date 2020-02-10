#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:24:41 2020

tf.keras多分类（mnist数据集）

@author: tinghai
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0

#%% mnist多分类：softmax交叉熵
(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
plt.imshow(train_image[0])
train_image = train_image/255.0 # 0-1值
test_image = test_image/255.0 # 0-1值

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # 将张量拉平成同一维度 (28,28) -> 28*28
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()
model.compile(opitimizer=tf.keras.optimizer.Adam(lr=0.01),
              loss='sparse_categorical_corssentropy', # 当label为顺序数字编码时
#              loss='categorical_corssentropy', # 当label为onehot编码时
              metrics=['acc'])
history = model.fit(train_image, train_label, epochs=30)
model.evaluate(test_image, test_label) # 评估
predict = model.predict(test_image) #预测
print(predict[0]) # 返回预测概率
print(np.argmax(predict[0])) # 返回最大概率的位置
print(test_label[0]) # 真实标签

train_label_hoehot = tf.keras.utils.to_categorial(train_label) # 转化train_label为onehot标签
test_label_hoehot = tf.keras.utils.to_categorial(test_label) # 转化test_label为onehot标签

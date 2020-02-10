#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:32:04 2020

tf.keras文本向量化

@author: tinghai
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0

#%% 文本向量化
from tensorflow import keras
from tensorflow.keras import layers

# 电影评论数据
data = keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = data.load_data(num_words=10000) #已经将文本转化成ID
#d = data.get_word_index()
#print(np.mean([len(x) for x in x_train])) # 238
x_train = keras.preprocessing.sequence.pad_sequences(x_train, 300) #填充0,使得长度为300
x_test = keras.preprocessing.sequence.pad_sequences(x_test, 300)

#test = 'i am a student ahh'
#[d[x] if x in d.keys() else 0 for x in test.split()]
#{x:d[x] for x in test.split() if x in d.keys()}

# 构建模型
model = keras.models.Sequential()
model.add(layers.Embedding(10000, 50, input_length=300)) #向量化，input_length输入数据的长度, (None, 300, 50)
model.add(layers.Flatten()) #将输入展平，不影响批量大小，(None, 15000)
model.add(layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))#添加L2正则化
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
              loss='binary_crossentropy',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=256, validation_data=(x_test, y_test))


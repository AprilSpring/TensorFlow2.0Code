#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:25:46 2020

tf.keras函数式API
    参考：https://blog.csdn.net/zkbaba/article/details/102595417

@author: tinghai
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0

#%% 函数式API构建模型
# 1）不同于tf.keras.Sequential()的构建模式，函数式API构建模型示例如下，compile和fit与之前类似
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(units=10)(x)
outputs = tf.keras.layers.Softmax()(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


# 2) 多输入、单输出
input1 = tf.keras.Input(shape=(28,28))
input2 = tf.keras.Input(shape=(28,28))

x1 = tf.keras.layers.Flatten()(input1)
x2 = tf.keras.layers.Flatten()(input2)
# or
# x1 = tf.keras.layers.GlobalAveragePooling2D()(input1)
# x2 = tf.keras.layers.GlobalAveragePooling2D()(input2)

x = tf.keras.layers.concatenate([x1,x2])

x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)

outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=[input1, input2], outputs=outputs)

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
              loss='sparse_categorical_corssentropy', 
              metrics=['acc'])



# 3) 单输入、多输出
inputs = tf.keras.Input(shape=(28,28))

x = tf.keras.layers.Flatten()(inputs)
# or
# x = tf.keras.layers.GlobalAveragePooling2D()(inputs)

x1 = tf.keras.layers.Dense(32, activation='relu')(x)
x2 = tf.keras.layers.Dense(32, activation='relu')(x)

out_1 = tf.keras.layers.Dense(10, activation='softmax', name='out_1')(x1)
out_2 = tf.keras.layers.Dense(1, activation='sigmoid', name='out_2')(x2)

model = tf.keras.Model(inputs=inputs, outputs=[out_1, out_2])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
              loss={'out_1':'sparse_categorical_corssentropy', 'out_2':'binary_corssentropy'}, 
              metrics=['acc'])


# 4）共享层网络
text_input_a = tf.keras.Input(shape=(None,), dtype='int32')
text_input_b = tf.keras.Input(shape=(None,), dtype='int32')

# Embedding for 1000 unique words mapped to 128-dimensional vectors
shared_embedding = tf.keras.layers.Embedding(1000, 128)

# We reuse the same layer to encode both inputs
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)

# two logistic predictions at the end
prediction_a = tf.keras.layers.Dense(1, activation='sigmoid', name='prediction_a')(encoded_input_a)
prediction_b = tf.keras.layers.Dense(1, activation='sigmoid', name='prediction_b')(encoded_input_b)

model = tf.keras.Model(inputs=[text_input_a, text_input_b],
                       outputs=[prediction_a, prediction_b])

tf.keras.utils.plot_model(model, to_file="./shared_model.png") # 绘制网络结构


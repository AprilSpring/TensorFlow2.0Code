#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:25:46 2020

tf.keras函数式API

@author: tinghai
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0

#%% 函数式API
# 1) 多输入、单输出
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



# 2) 单输入、多输出
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




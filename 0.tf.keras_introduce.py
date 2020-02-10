#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:22:08 2020

简单介绍tf.keras

@author: tinghai
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0


#%% tf.keras
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,))) # 第一个参数是输出维度
model.summary()
model.compile(optimizer='adam', loss='mse')
model.fit(train_x, train_y, epochs=30)
model.predict(text_x)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:47:42 2020

使用LSTM预测时序问题

@author: tinghai
"""

#%% RNN - 北京空气污染预测
# 目标：预测未来的PM2.5水平

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
%matplotlib inline
import numpy as np
import pandas as pd
import re
import datetime

#%% 1）数据处理
data = pd.read_csv('./dataset/PRSA_data.csv') #label is PM2.5

# Nan数据处理
data['pm2.5'].isna().sum() #2067, 由于去掉PM2.5为空的序列会打乱数据的时序性，因此采取填充的方式
data = data.iloc[24:] #去掉前24个pm2.5为空的数据
data = data.fillna(method='ffill') #前向填充，使用前天的数据填充昨天的数据

# 时间列合并成一个索引值
data['tm'] = data.apply(lambda x: datetime.datetime(year=x['year'], 
                                       month=x['month'], 
                                       day=x['day'], 
                                       hour=x['hour']), axis=1)
data.drop(columns=['year','month','day','hour','No'], inplace=True)
data = data.set_index('tm')

# 非数字列处理
print(data.cbwd.unique())
data = data.join(pd.get_dummies(data.cbwd)) #cbwd列进行onehot编码后，与原dataframe进行拼接
del data.cbwd

# 数据时序性采样
# 使用前面多久的时序数据，来预测接下来多久的数据？
data['pm2.5'][-1000:].plot() # 最后1000次PM2.5观测情况

seq_length = 5 * 24 #使用当前点前5天的数据
delay = 24 #预测当前点后1天的数据，因此delay=24h

data_ = []
for i in range(len(data) - seq_length - delay):
    data_.append(data.iloc[i: i + seq_length + delay]) #按时序依次采样6天的数据

data_ = np.array([df.values for df in data_]) #转化成numpy形式
print(data_.shape) #(43656,144,11)，一共采样出43656条时序数据，每条时序包括144个时间点，每个时间点包含11个特征值

# 训练和测试数据生成
np.random.shuffle(data_)
x = data_[:,:seq_length,:] #(43656,120,11)
y = data_[:,-1,0] #(43656,)，-1 => 取每条时序的最后一个时间点，0 => 最后一个时间点的pm2.5值
split_b = int(0.8 * data_.shape[0])
train_x = x[:split_b]
train_y = y[:split_b]
test_x = x[split_b:]
test_y = y[split_b:]

# 数据标准化 
# 注意：a. 训练集需要单独进行标准化，不能使用全局数据的标准化结果
# b. 测试集也应该使用训练集的均值和标准差进行标准化
# c. 预测值是否需要进行标准化呢？不需要
mean = train_x.mean(axis=0) #按列计算均值
std = train_x.std(axis=0) #按列计算标准差
train_x = (train_x - mean)/std
test_x = (test_x - mean)/std

#%% 2）构建全连接神经网络
BATCH_SIZE = 32
model = tf.keras.Sequentail()
model.add(tf.keras.Flatten(input_shape=train_x.shape[1:]))
model.add(tf.keras.Dense(32,activation='relu'))
model.add(tf.keras.Dense(1)) #回归问题，无需激活函数
model.compile(optimizer='adam', 
              loss='mse', 
              metrics=['mae'])
history = model.fit(train_x,train_y, 
                    epochs=50, 
                    steps_per_epoch=split_b//BATCH_SIZE,
                    validation_data=(test_x,test_y), 
                    validation_steps=(data_.shape[0]-split_b)//BATCH_SIZE)
plt.plot(history.epoch, history.history['mean_absolute_error'], 'r', label='train-loss')
plt.plot(history.epoch, history.history['val_mean_absolute_error'], 'g', label='validation-loss')
plt.legend()

#%% 3）构建单层LSTM网络
model = tf.keras.Sequentail()
model.add(tf.keras.layers.LSTM(units=32,
                               input_shape=train_x.shape[1:],
                               activation='tanh')) #默认return_sequences=False, 即只返回LSTM的output结果，而非state结果
model.add(tf.keras.Dense(1))
model.compile(optimizer='adam', 
              loss='mse', 
              metrics=['mae'])
history = model.fit(train_x,train_y, 
                    epochs=150, 
                    steps_per_epoch=split_b//BATCH_SIZE,
                    validation_data=(test_x,test_y), 
                    validation_steps=(data_.shape[0]-split_b)//BATCH_SIZE)
plt.plot(history.epoch, history.history['mean_absolute_error'], 'r', label='train-loss')
plt.plot(history.epoch, history.history['val_mean_absolute_error'], 'g', label='validation-loss')
plt.legend()

#%% 4）构建多层LSTM网络
model = tf.keras.Sequentail()
model.add(tf.keras.layers.LSTM(units=32,
                               input_shape=train_x.shape[1:],
                               activation='tanh',
                               return_sequences=True)) #返回LSTM的output和state结果
model.add(tf.keras.layers.LSTM(units=32,
                               activation='tanh',
                               return_sequences=True))
model.add(tf.keras.layers.LSTM(units=32,
                               activation='tanh',
                               return_sequences=False)) #最后一个LSTM，只使用output输出，用于连接Dense层
model.add(tf.keras.Dense(1))

# 添加回调函数，在训练过程中降低学习速率
#在连续3个epoch中val_loss没有降低，则降低LR为原来的0.1倍，但最小不超过0.00001
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 patience=3, 
                                                 factor=0.1, 
                                                 min_lr=0.00001)
model.compile(optimizer='adam', 
              loss='mse', 
              metrics=['mae'])
history = model.fit(train_x,train_y,
                    epochs=200, 
                    steps_per_epoch=split_b//BATCH_SIZE,
                    validation_data=(test_x,test_y), 
                    validation_steps=(data_.shape[0]-split_b)//BATCH_SIZE,
                    callbacks=[lr_reduce])
plt.plot(history.epoch, history.history['mean_absolute_error'], 'r', label='train-loss')
plt.plot(history.epoch, history.history['val_mean_absolute_error'], 'g', label='validation-loss')
plt.legend()


# 模型评估
model.evaluate(test_x, test_y, verbose=0)

# 测试集预测
pred_y = model.predict(test_x)

# 新数据预测
data_test = data[-120:]
data_test = data_test.iloc[:,5:]
data_test = data_test.join(pd.get_dummies(data_test['cbwd']))
data_test = data_test.drop('cbwd',axis=1, inplace=True)
data_test.reindex(columns=train_x.columns)
data_test = (data_test - mean)/std # 使用训练数据的均值和方差，对预测数据进行归一化
data_test = data_test.to_numpy() #转化成array
data_test = np.expand_dims(data_test, axis=0) #扩展第一个维度，即batch
data_y = model.predict(data_test)
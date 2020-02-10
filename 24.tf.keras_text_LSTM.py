#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:33:12 2020

LSTM预测航空公司评论

@author: tinghai
"""

#%% RNN
# tf.keras.layers.LSTM
# tf.keras.layers.GRU
# input_shape is (batch_size, seq_length, embedding_size)
# output_shape is (batch_size, tags)

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
%matplotlib inline
import numpy as np
import re

# 数据集：航空公司评论数据集
data = pd.read_csv('./Tweet.csv') #including text and label
data = data[['airline_sentiment','text']]
print(data.airline_sentiment.unique()) #标签种类
print(data.airline_sentiment.value_count()) #样本分布
data_p = data[data.airline_sentiment=='positive']
data_n = data[data.airline_sentiment=='negative']
data_n = data_n.iloc[:data_p.shape[0]]
data = pd.concat([data_p,data_n])
data = data.sample(len(data))
data['label'] = (data.airline_sentiment == 'positive').astype(int)
del data['airline_sentiment']

# 文本清洗
def reg_text(text):
    token = re.compile('[A-Za-z]+|[!?,.()]')
    new_text = token.findall(text)
    new_text = [word.lower() for word in new_text]
    return new_text

data['text'] = data['text'].apply(reg_text)

# 文本转ID
word_set = list(set([word for word in text for text in data['text'].tolist()]))
word_index = {}
for index, word in enumerate(word_set):
    word_index[word] = index + 1 #由于使用0进行填充，因此index从1开始
data_ok = data['text'].apply(lambda x: [word_index.get(word,0) for word in x])

# 文本长度分析
text_len = data_ok.apply(lambda x : len(x))
print(text_len.describe())
max_len = max(text_len) #40
max_word = len(word_set) + 1 #1为填充
data_ok = tf.keras.preprocessing.sequence.pad_sequences(data_ok.values, max_len) #填充0,使得长度为max_len

# LSTM模型构建
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(max_word, 50, input_length=max_len)) #向量化，input_length输入数据的长度, (None, max_len, 50)
model.add(tf.keras.layers.LSTM(64))
#model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))) #添加L2正则化
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
              loss='binary_crossentropy',
              metrics=['acc'])
model.fit(data_ok, data.labels.values, epochs=10, batch_size=32, validation_split=0.2) #选择20%作为测试集  


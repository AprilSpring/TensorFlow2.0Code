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
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0


#%% tf.keras使用
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,))) # 第一个参数是输出维度
model.summary()
model.compile(optimizer='adam', loss='mse')
model.fit(train_x, train_y, epochs=30)
model.predict(text_x)


#%% 1）model.fit()方法
# fit()函数传入的x_train和y_train是被完整的加载进内存的
# 参数详情
# tf.keras.models.fit(
    # self,
    # x=None, #训练数据
    # y=None, #训练数据label标签
    # batch_size=None, #每经过多少个sample更新一次权重，defult 32
    # epochs=1, #训练的轮数epochs
    # verbose=1, #0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
    # callbacks=None, #list，list中的元素为keras.callbacks.Callback对象，在训练过程中会调用list中的回调函数
    # validation_split=0., #浮点数0-1，将训练集中的一部分比例作为验证集，然后下面的验证集validation_data将不会起到作用
    # validation_data=None, #验证集
    # shuffle=True, #布尔值和字符串，如果为布尔值，表示是否在每一次epoch训练前随机打乱输入样本的顺序，如果为"batch"，为处理HDF5数据
    # class_weight=None, #dict,分类问题的时候，有的类别可能需要额外关注，分错的时候给的惩罚会比较大，所以权重会调高，体现在损失函数上面
    # sample_weight=None, #array,和输入样本对等长度,对输入的每个特征+个权值，如果是时序的数据，则采用(samples，sequence_length)的矩阵
    # initial_epoch=0, #如果之前做了训练，则可以从指定的epoch开始训练
    # steps_per_epoch=None, #将一个epoch分为多少个steps，也就是划分一个batch_size多大，比如steps_per_epoch=10，则就是将训练集分为10份，不能和batch_size共同使用
    # validation_steps=None, #当steps_per_epoch被启用的时候才有用，验证集的batch_size
    # **kwargs #用于和后端交互
# )
# 返回的是一个History对象，可以通过History.history来查看训练过程，loss值等等


#%% 2）model.fit_generator()方法
# 当训练数据量很大，那么是不可能将所有数据载入内存的，必将导致内存泄漏，这时候我们可以用fit_generator函数来进行训练
# generator参数：使用 Python 生成器（或 Sequence (tf.keras.utils.Sequence) 实例）
# 逐批生成的数据，按批次训练模型。生成器与模型并行运行，以提高效率。

# 例如，这可以让你在 CPU 上对图像进行实时数据增强，以在 GPU 上训练模型。
class Generator():
    def __init__(self, X, y, batch_size=32, aug=False):
        def generator():
            idg = ImageDataGenerator(horizontal_flip=True,
                                     rotation_range=20,
                                     zoom_range=0.2)
            while True:
                for i in range(0, len(X), batch_size):
                    X_batch = X[i:i+batch_size].copy()
                    y_batch = [x[i:i+batch_size] for x in y]
                    if aug:
                        for j in range(len(X_batch)):
                            X_batch[j] = idg.random_transform(X_batch[j])
                    yield X_batch, y_batch
        self.generator = generator()
        self.steps = len(X) // batch_size + 1

gen_train = Generator(train_x, train_y, batch_size=32, aug=True)

model = Sequential()
model.add(Dense(units=1000, activation='relu', input_dim=2))
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit_generator(generator=gen_train.generator,
                    steps_per_epoch=gen_train.steps,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    validation_data=(valid_x, valid_y),
                    validation_steps=None,
                    class_weight=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    shuffle=True,
                    initial_epoch=0)



#%% model.train_on_batch()
model.train_on_batch(batchX, batchY)







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:34:30 2020

eager自定义训练：mnist数据集

@author: tinghai
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0


#%% eager自定义训练模式 - minst示例
'''
步骤：
    1)按Batch准备数据
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(10000).batch(32)
    2)定义模型结构
        model = tf.keras.Sequential([...])
    3)选择optimizer
        optimizer = tf.keras.optimizer.Adam()
    4)计算loss
        y_ = model(x)
        loss = loss_func(y,y_)
    5)计算grads
        with tf.GradientTape() as t
        grads = t.gradient(loss, model.trainable_variables)
    6)optimizer按照grads方向更新参数
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    7)按batch进行训练
        重复 5)和 6）
'''

# 生成数据集
(train_image, train_labels), (test_image, test_labels) = tf.keras.datasets.mnist.load_data()

# 训练集
train_image = tf.expand_dims(train_image, -1) #-1表示扩增的最后一个维度，由于使用CNN因此需要扩增数据维度
train_image = tf.cast(train_image/255, tf.float32) #需要float类型才能做梯度运算
train_labels = tf.cast(train_labels, tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels))
dataset = dataset.shuffle(10000).batch(32) # 默认repeat(1)；如果使用fit方法的话，需添加repeat(),无限循环

# 测试集
test_image = tf.expand_dims(test_image, -1)
test_image = tf.cast(test_image/255, tf.float32)
test_labels = tf.cast(test_labels, tf.int64)
test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_labels))
test_dataset = test_dataset.batch(32)

# 模型构建
model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16,[3,3], activation='relu',input_shape=(28,28,1)), #任意图片大小：input_shape=(None,None,1)
        tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
        tf.keras.layser.GlobalMaxPooling2D(), #GlobalAveragePooling2D()
        tf.keras.layers.Dense(10, activation='softmax')])

# 自定义模型优化（不使用compile）
optimizer = tf.keras.optimizer.Adam(lr=0.01) #初始化优化器
#loss_func = tf.keras.losses.sparse_categorial_crossentropy(y_true, y_pred, from_logits = False) #是否从上层Dense激活，如果是则True，否则False
# or
loss_func = tf.keras.losses.SparseCategorialCrossentropy(from_logits=False) #返回一个方法，loss_func(y, y_)

features, labels = next(iter(dataset)) #按照batch迭代返回数据
predictions = model(features) #计算预测结果
print(predictions.shape) # (32, 10)
tf.argmax(predictions, axis=1) #同np.argmax(), 返回预测概率最大的位置

# 计算loss
def loss(model, x, y):
    y_ = model(x)
#    y_ = tf.argmax(y_, axis=1) # 不需要吗？不需要！
    loss = tf.keras.losses.SparseCategorialCrossentropy(from_logits=False)(y, y_) #if loss_func is SparseCategorialCrossentropy
#    loss = tf.keras.losses.sparse_categorial_crossentropy(y, y_, from_logits=False) #if loss_func is sparse_categorial_crossentropy
    return loss

# 评估指标
train_loss = tf.keras.metrics.Mean('train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

test_loss = tf.keras.metrics.Mean('test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

# 每个batch的训练
def train_step(model, images, labels):
    with tf.GradientTape() as t:
        predictions = model(images)
        loss_step = loss_func(labels, predictions)
#        loss_step = loss(model, images, labels) #计算loss
    grads = t.gradient(loss_step, model.trainable_variables) #计算loss相对模型变量的梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables)) #使用grads更新模型变量，即优化过程
    train_loss(loss_step) #计算平均loss，备注：在循环过程中会记录下每个Batch的loss
    train_accuracy(labels, predictions) #计算平均accuracy

# 每个batch的预测（不用计算grads和optimizer）
def test_step(model, images, labels):
    with tf.GradientTape() as t:
        pred = model(images)
        loss_step = loss_func(labels, pred)
    test_loss(loss_step)
    test_accuracy(labels, predictions)

# 训练    
def train():
    for epoch in range(10):
        # 训练
        for (batch, (images, labels)) in enumerate(dataset):
            train_step(model, images, labels) #every batch
        print('Epoch{} is finished. loss is {}, accuracy is {}.' \
              .format(epoch, train_loss.result(), train_accuracy.result()))
        # 预测
        for (batch, (images, labels)) in enumerate(test_dataset):
            test_step(model, images, labels) #every batch
        print('Epoch{} is finished. test_loss is {}, test_accuracy is {}.' \
              .format(epoch, test_loss.result(), test_accuracy.result()))
        
        # 重制状态
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


train() #训练模型

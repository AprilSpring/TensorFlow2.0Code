#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:38:50 2020

eager模式中tensorboard使用：mnist数据集

@author: tinghai
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0

#%% eager自定义训练中的tensorboard
(train_image, train_labels), (test_image, test_labels) = tf.keras.datasets.mnist.load_data()
train_image = tf.expand_dims(train_image, -1) #-1表示扩增的最后一个维度，由于使用CNN因此需要扩增数据维度
train_image = tf.cast(train_image/255, tf.float32) #需要float类型才能做梯度运算
train_labels = tf.cast(train_labels, tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels))
dataset = dataset.shuffle(10000).repeat().batch(32) # 默认repeat(1)；如果使用fit方法的话，需添加repeat(),无限循环

test_image = tf.expand_dims(test_image, -1)
test_image = tf.cast(test_image/255, tf.float32)
test_labels = tf.cast(test_labels, tf.int64)
test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_labels))
test_dataset = test_dataset.batch(32)

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16,[3,3], activation='relu',input_shape=(28,28,1)), #任意图片大小：input_shape=(None,None,1)
        tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
        tf.keras.layser.GlobalMaxPooling2D(), #GlobalAveragePooling2D()
        tf.keras.layers.Dense(10, activation='softmax')])

optimizer = tf.keras.optimizer.Adam(lr=0.01) #初始化优化器
loss_func = tf.keras.losses.SparseCategorialCrossentropy(from_logits=False) #返回一个方法，loss_func(y, y_)

train_loss = tf.keras.metrics.Mean('train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

test_loss = tf.keras.metrics.Mean('test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

def train_step(model, images, labels):
    with tf.GradientTape() as t:
        predictions = model(images)
        loss_step = loss_func(labels, predictions)
    grads = t.gradient(loss_step, model.trainable_variables) #计算loss相对模型变量的梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables)) #使用grads更新模型变量，即优化过程
    train_loss(loss_step) #计算平均loss，备注：在循环过程中会记录下每个Batch的loss
    train_accuracy(labels, predictions) #计算平均accuracy

def test_step(model, images, labels):
    with tf.GradientTape() as t:
        pred = model(images)
        loss_step = loss_func(labels, pred)
    test_loss(loss_step)
    test_accuracy(labels, predictions)

current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

train_writer = tf.summary.create_file_writer(train_log_dir)
test_writer = tf.summary.create_file_writer(test_log_dir)

  
def train():
    for epoch in range(10):
        print('Epoch is {}'. format(epoch))
        # 训练
        for (batch, (images, labels)) in enumerate(dataset):
            train_step(model, images, labels) #every batch
        with train_writer.set_as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('acc', train_accuracy.result(), step=epoch)
        print('train_end')
        
        # 预测
        for (batch, (images, labels)) in enumerate(test_dataset):
            test_step(model, images, labels) #every batch
        with test_writer.set_as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('acc', test_accuracy.result(), step=epoch)
        print('test_end')
        
        # 重制状态
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


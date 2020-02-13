#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:28:49 2020

@tf.function使用
    a. 用于转换成Tensorflow的计算图，尤其在模型部署时需要用到 (转化为pb模型时)
    b. 只需要将我们希望以 Graph Execution 模式运行的代码封装在一个函数内，并在函数前加上 @tf.function 即可

@author: tinghai
"""

#%% 自动图运算（Graph execution）
# 使用@tf.function装饰器，实现自动图运算，从而将模型转换为易于部署的tensorflow图模型
# 内部机制：
        #在eager模型关闭下，函数内代码依次运行，每个tf代码都只定义了计算节点，而非真正的计算
        #使用AutoGraph将函数中的python控制流转换成Tensorflow计算图中对应节点，比如while,for转换为tf.while，if转换为tf.cond等
        #建立函数内代码计算图，为了保证计算图的顺序，图中还会自动添加一些tf.control_dependencies节点
        #运行一次该计算图
        #基于函数类型和输入函数参数类型生成一个哈希值，并将建立的计算图缓存到一个哈希表中
        #在被@tf.function修饰的函数被再次调用时，根据函数名和输入的函数参数类型计算哈希值，检查哈希表中是否有对应计算图的缓存，如果是则继续使用已缓存的计算图，否则的话根据上述步骤建立计算图。
# 使用方法：
        #当定义多个函数实现不同运算式时，仅需要在最后调用的函数上添加@tf.function即可，这样所有的运算节点都会被编译。

# 参考：https://blog.csdn.net/zkbaba/article/details/103915132

import tensorflow as tf
import time
from zh.model.mnist.cnn import CNN
from zh.model.utils import MNISTLoader

num_batches = 400
batch_size = 50
learning_rate = 0.001
data_loader = MNISTLoader()

@tf.function
def train_one_step(X, y):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
    loss = tf.reduce_mean(loss)
    # 注意这里使用了TensorFlow内置的tf.print()。@tf.function不支持Python内置的print方法
    tf.print("loss", loss)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

if __name__ == '__main__':
    model = CNN()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    start_time = time.time()
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        train_one_step(X, y)
    end_time = time.time()
    print(end_time - start_time)


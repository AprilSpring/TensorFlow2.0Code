#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:28:49 2020

@tf.function使用

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

@tf.function
def train_step(model, images, labels):
    pass



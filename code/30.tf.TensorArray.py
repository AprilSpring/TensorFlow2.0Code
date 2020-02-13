#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Created on: 2020/2/13 3:16 下午

@annotaion:
    tf.TensorArray：动态数组
    用于：计算图模式中使用动态数组保存和读取张量
    参考：https://blog.csdn.net/zkbaba/article/details/104101553

@author: tinghai
"""

import tensorflow as tf

# arr = tf.TensorArray(dtype, size, dynamic_size=False) #dynamic_size=True，则该数组会自动增长空间
# arr = arr.write(index, value) # 将 value 写入数组的第 index 个位置
# arr_1 = read(index) #读取数组的第 index 个值

@tf.function
def array_write_and_read():
    arr = tf.TensorArray(dtype=tf.float32, size=3)
    arr = arr.write(0, tf.constant(0.0))
    arr = arr.write(1, tf.constant(1.0))
    arr = arr.write(2, tf.constant(2.0))
    arr_0 = arr.read(0)
    arr_1 = arr.read(1)
    arr_2 = arr.read(2)
    return arr_0, arr_1, arr_2

a, b, c = array_write_and_read()
print(a, b, c)








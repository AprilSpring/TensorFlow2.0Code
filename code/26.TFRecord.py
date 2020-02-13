#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Created on: 2020/2/12 12:16 下午

@annotaion:
    TFRecord学习
    原文链接：https://blog.csdn.net/zkbaba/article/details/103433011

@author: tinghai
"""

# 1）TFRecord介绍
# TFRecord 是 TensorFlow 中的数据集存储格式
# TFRecord 可以理解为一系列序列化的 tf.train.Example 元素所组成的列表文件
# 每一个 tf.train.Example 又由若干个 tf.train.Feature 的字典组成

# dataset.tfrecords
[
    {   # example 1 (tf.train.Example)
        'feature_1': tf.train.Feature,
        ...
        'feature_k': tf.train.Feature
    },
    ...
    {   # example N (tf.train.Example)
        'feature_1': tf.train.Feature,
        ...
        'feature_k': tf.train.Feature
    }
]


# 2）TFRecord生成
# a.建立 tf.train.Feature 字典
    # tf.train.Feature 支持以下三种数据格式：
    # tf.train.BytesList：字符串或原始 Byte 文件（如图片），通过bytes_list参数传入一个由字符串数组初始化的 tf.train.BytesList 对象；
    # tf.train.FloatList：浮点数，通过float_list参数传入一个由浮点数数组初始化的 tf.train.FloatList 对象；
    # tf.train.Int64List：整数，通过int64_list参数传入一个由整数数组初始化的 tf.train.Int64List 对象
# b.建立 tf.train.Example 对象
# c.写入 TFRecord 文件
import tensorflow as tf
import os

data_dir = './cats_vs_dogs'
train_cats_dir = data_dir + '/train/cats/'
train_dogs_dir = data_dir + '/train/dogs/'
tfrecord_file = data_dir + '/train/train.tfrecords'

train_cat_filenames = [train_cats_dir + filename for filename in os.listdir(train_cats_dir)]
train_dog_filenames = [train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)]
train_filenames = train_cat_filenames + train_dog_filenames
train_labels = [0] * len(train_cat_filenames) + [1] * len(train_dog_filenames)  # 将 cat 类的标签设为0，dog 类的标签设为1

# 迭代读取每张图片，建立 tf.train.Feature 字典和 tf.train.Example 对象，序列化并写入 TFRecord 文件
with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for filename, label in zip(train_filenames, train_labels):
        image = open(filename, 'rb').read()     # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
        feature = {                             # 建立 tf.train.Feature 字典
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))   # 标签是一个 Int 对象
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example
        writer.write(example.SerializeToString())   # 将Example序列化并写入 TFRecord 文件



# 3）TFRecord读取
# tf.io.FixedLenFeature 的三个输入参数 shape,dtype,default_value（可省略）为每个 Feature 的形状、类型和默认值
# 使用 tf.io.parse_single_example 函数对数据集中的每一个序列化的 tf.train.Example 对象解码
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)    # 读取 TFRecord 文件

feature_description = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

def _parse_example(example_string): # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])    # 解码JPEG图片
    return feature_dict['image'], feature_dict['label']

dataset = raw_dataset.map(_parse_example)

import matplotlib.pyplot as plt
for image, label in dataset:
    plt.title('cat' if label == 0 else 'dog')
    plt.imshow(image.numpy()) #imshow()基于图片解码后的数据生成图片
    plt.show()







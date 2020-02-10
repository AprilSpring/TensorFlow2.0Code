#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:52:17 2020

预训练模型fine-tune方法

@author: tinghai
"""

#%% Fine-tune
# 冻结预训练模型底层卷积层参数、共同训练顶层卷积层和新添加的顶层全连接层参数
# 步骤：
# 1）在预训练模型上添加顶层全连接层和输出层
# 2）冻结预训练模型的所有参数
# 3）训练新添加的分类层参数
# 4）解冻预训练模型的部分参数（比如靠上的几层）
# 5）联合训练解冻的卷积层和新添加的自定义层

import tensorflow as tf
from tensorflow import keras    
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np 
import glob
import os

tf.test.is_gpu_avialble() #True is used GPU
keras = tf.keras
layers = tf.keras.layers

# 图片处理
def load_preprocess_image(path, label):
    image = tf.io.read_file(path) #读取图片
    image = tf.image.decode_jpeg(image, channels=3) #解码图片
    image = tf.image.resize(image, (256, 256)) #转换所有图片大小相同
    image = tf.cast(image, tf.float32) #转换数据为float类型
    image = image/255 #归一化
    label = tf.reshape(label,[1]) #[1,2,3] => [[1],[2],[3]]
    return image, label

BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE #根据CPU数目，自动使用并行运算

# train数据进行数据增强
train_image_path = glob.glob('./train/*/*.jpg') # * is cat/ or dog/
train_image_label = [int(x.split('/')[2] == 'cats') for x in train_image_path] # 0-cat, 1-dog
train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
train_image_ds = train_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
train_count = len(train_image_path)
train_image_ds = train_image_ds.shuffle(train_count).batch(BATCH_SIZE)
train_image_ds = train_image_ds.prefetch(AUTOTUNE) #在前台已读取数据的训练同时，预先读取后台数据

# test数据处理不需要增强
test_image_path = glob.glob('./test/*.jpg')
test_image_label = [int(x.split('/')[2] == 'cats') for x in test_image_path] # 0-cat, 1-dog
test_image_ds = tf.data.Dataset.from_tensor_slices((testimage_path, test_image_label))
test_image_ds = test_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE) #num_parallel_calls并行运算CPU数目
test_count = len(test_image_path)
test_image_ds = test_image_ds.batch(BATCH_SIZE)
test_image_ds = test_image_ds.prefetch(AUTOTUNE) #在前台已读取数据的训练同时，预先读取后台数据

# 按batch取出数据进行查看
imgs, labels = next(iter(train_image_ds))
print(imgs) #(32, 256, 256, 3)
plt.imshow(imgs[0]) #显示图片

# 1）在预训练模型基础上，添加顶层全连接层和输出层
covn_base = keras.applications.VGG16(weight='imagenet', #weight=None 不使用预训练模型网络参数
                                     include_top=False) #include_top=False 不使用顶层全连接层的参数
                                         #权重文件存放：/Users/tinghai/.keras/models
covn_base.summary()

model = keras.Sequential()
model.add(covn_base)
model.add(layers.GlobalAveragePooling2D()) #类似于Flatten()，将张量展开好接全连接层
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# 2）冻结预训练模型的所有参数
covn_base.trainable = False
model.summary() #可训练的参数明显减少

# 3）训练新添加的分类层参数
model.compile(optimizer=keras.optimizers.Adam(lr=0.0005),
              loss='binary_corssentropy',
              metrics=['acc'])
history = model.fit(train_image_ds,
                    steps_per_epoch=train_count//BATCH_SIZE,
                    epochs=12,
                    validation_data=test_image_ds,
                    validation_steps=test_count//BATCH_SIZE)

# 4）解冻预训练模型的部分参数
covn_base.trainable = True
len(covn_base.layers) #预训练模型一共19层

fine_tune_at = -3
for layer in covn_base.layers[:fine_tune_at]:
    layer.trainable = False #除去后3层，其余都是不可训练的

# 5）联合训练解冻的预训练层和新添加的自定义层
model.compile(optimizer=keras.optimizers.Adam(lr=0.0005/10), #需要使用更小的lr
              loss='binary_corssentropy',
              metrics=['acc'])

initial_epochs = 12
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history = model.fit(train_image_ds,
                    steps_per_epoch=train_count//BATCH_SIZE,
                    epochs=total_epochs,
                    initial_epoch = initial_epochs, #新增参数
                    validation_data=test_image_ds,
                    validation_steps=test_count//BATCH_SIZE)


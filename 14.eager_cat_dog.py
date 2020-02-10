#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:40:24 2020

猫狗数据集自定义分类模型和可视化

@author: tinghai
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0

#%% 猫狗数据自定义训练示例
import tensorflow as tf
from tensorflow import keras    
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np 
import glob
import os

# 1) reload data to tf.data
# 图片处理
def load_preprocess_image(path, label):
    image = tf.io.read_file(path) #读取图片
    image = tf.image.decode_jpeg(image, channels=3) #解码图片
    image = tf.image.resize(image, (256, 256)) #转换所有图片大小相同
    image = tf.cast(image, tf.float32) #转换数据为float类型
    image = image/255 #归一化
#    image = tf.image.convert_image_dtype(image) #如果原数据不是float类型，会默认把数据做归一化；如果原数据是float类型，则不会进行归一化
    label = tf.reshape(label,[1]) #[1,2,3] => [[1],[2],[3]]
    return image, label

# 图片增强
# 针对训练数据进行增强：比如上下翻转、左右翻转、图片裁剪等
def load_preprocess_image_enhance(path, label):
    image = tf.io.read_file(path) #读取图片
    image = tf.image.decode_jpeg(image, channels=3) #解码图片
    image = tf.image.resize(image, (360, 360)) #转换所有图片大小相同
    image = tf.image.random_crop(image, [256, 256, 3]) #讲360*360的图像随机裁剪为256*256
    image = tf.image.random_flip_left_right(image) #左右翻转
    image = tf.image.random_flip_up_down(image) #上下翻转
#    image = tf.image.random_brigtness(image, 0.5) #随机改变亮度
#    image = tf.image.random_contrast(image, 0, 1) #随机改变对比度
#    image = tf.image.random_hue(image, max_delta=0.3) #随机改变颜色
#    image = tf.image.random_saturation(image, lower=0.2, upper=1.0) #随机改变饱和度
    image = tf.cast(image, tf.float32) #转换数据为float类型
    image = image/255 #归一化
#    image = tf.image.convert_image_dtype(image) #如果原数据不是float类型，会默认把数据做归一化；如果原数据是float类型，则不会进行归一化
    label = tf.reshape(label,[1]) #[1,2,3] => [[1],[2],[3]]
    return image, label

BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE #根据CPU数目，自动使用并行运算

# train数据进行数据增强
train_image_path = glob.glob('./train/*/*.jpg') # * is cat/ or dog/
train_image_label = [int(x.split('/')[2] == 'cats') for x in train_image_path] # 0-cat, 1-dog
train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
#train_image_ds = train_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE) #num_parallel_calls并行运算CPU数目
train_image_ds = train_image_ds.map(load_preprocess_image_enhance, num_parallel_calls=AUTOTUNE)

# 取出一张图片查看
for img, label in train_image_ds.take(1):
    plt.imshow(img)
    
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

# 2) model construction
model = keras.Sequential([
        tf.keras.layers.Conv2D(64,(3,3),input_shape=(256,256,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(512,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(1024,(3,3),activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(), #(None,1024)
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])
model.summary()

#pred = model(imags) #(32, 1)
#y_ = np.array([p[0].numpy() for p in tf.cast(pred>0.5, tf.int32)]) # pred>0.5返回boolen值
#y = np.array([l[0].numpy() for l in labels])

# 3) define loss and optimizer
ls = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizer.Adam(lr=0.01)

# 4) train define (for one batch)
train_epoch_loss_avg = tf.keras.metrics.Mean('train_loss')
train_accuracy = tf.keras.metrics.Accuracy('train_acc')
def train_step(model, images, labels):
    with tf.GradientTape() at t:
        pred = model(images)
        loss_step = ls(labels, pred)
    grads = t.gradient(loss_step, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_epoch_loss_avg(loss_step)
    train_accuracy(labels, tf.cast(pred>0.5, tf.int32))

# 5) test define (for one batch)
test_epoch_loss_avg = tf.keras.metrics.Mean('test_loss')
test_accuracy = tf.keras.metrics.Accuracy('test_acc')
def test_step(model, images, labels):
    pred = model(images, training=False)
    # or
    # pred = model.predict(images)
    loss_step = ls(labels, pred)
    test_epoch_loss_avg(loss_step)
    test_accuracy(labels, tf.cast(pred>0.5, tf.int32))

# 6) model training and validation
train_loss_results = []
train_acc_results = []
test_loss_results = []
test_acc_results = []
num_epochs = 30
for epoch in range(num_epochs):
    # train
    for (batch, (imgs_, labels_)) in train_image_ds:
        train_step(model, images=imgs_, labels=labels_)
        print('.', end='')
    print()
    train_loss_results.append(train_epoch_loss_avg.result())
    train_acc_results.append(train_accuracy.result())
    
    # test
    for (batch, (imgs_, labels_)) in test_image_ds:
        test_step(model, images=imgs_, labels=labels_)
        print('.', end='')
    print()
    test_loss_results.append(test_epoch_loss_avg.result())
    test_acc_results.append(test_accuracy.result())
    
    # print
    print('Epoch:{}, train_loss:{:.3f}, train_accuracy:{:.3f}, test_loss:{:.3f}, test_accuracy:{:.3f}'.\
          format(epoch+1, 
                 train_epoch_loss_avg.result(), 
                 train_accuracy.result(), 
                 test_epoch_loss_avg.result(), 
                 test_accuracy.result()))
    
    # reset
    train_epoch_loss_avg.reset_states()
    train_accuracy.reset_states()
    test_epoch_loss_avg.reset_states()
    test_accuracy.reset_states()

# 7) model optimization
# 增加网络的深度（增加卷积层、全连接层），且避免过拟合（数据增强、添加batch normalization、添加dropout）
# 类似VGG16模型: 2个64-conv2D + 2个128-conv2D + 3个256-conv2D + 3个512-conv2D + 3个512-conv2D
model = keras.Sequential([
        tf.keras.layers.Conv2D(64,(3,3),input_shape=(256,256,3), activation='relu'),
        tf.keras.layers.Batchnormalization(), # 放在卷积层后
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.Batchnormalization()
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
        tf.keras.layers.Batchnormalization()
        tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
        tf.keras.layers.Batchnormalization()
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Conv2D(256,(3,3),activation='relu'),
        tf.keras.layers.Batchnormalization()
        tf.keras.layers.Conv2D(256,(3,3),activation='relu'),
        tf.keras.layers.Batchnormalization()
        tf.keras.layers.Conv2D(256,(3,3),activation='relu'),
        tf.keras.layers.Batchnormalization()
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Conv2D(512,(3,3),activation='relu'),
        tf.keras.layers.Batchnormalization()
        tf.keras.layers.Conv2D(512,(3,3),activation='relu'),
        tf.keras.layers.Batchnormalization()
        tf.keras.layers.Conv2D(512,(1,1),activation='relu'), #1*1卷积，用于提取channel
        tf.keras.layers.Batchnormalization()
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Conv2D(512,(3,3),activation='relu'),
        tf.keras.layers.Batchnormalization()
        tf.keras.layers.Conv2D(512,(3,3),activation='relu'),
        tf.keras.layers.Batchnormalization()
        tf.keras.layers.Conv2D(512,(1,1),activation='relu'), #1*1卷积，用于提取channel
        tf.keras.layers.Batchnormalization()
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.GlobalAveragePooling2D(), #(None,512)
        
        tf.keras.layers.Dense(4096, activation='relu'), #全连接层1
        tf.keras.layers.Dense(4096, activation='relu'), #全连接层2
        tf.keras.layers.Dense(1000, activation='relu'), #全连接层3
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])
model.summary()


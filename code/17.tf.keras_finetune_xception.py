#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:56:47 2020

Xception预训练模型的fine-tune训练

@author: tinghai
"""

#%% Fine-tune
# 冻结预训练模型底层卷积层参数、共同训练顶层卷积层和新添加的顶层全连接层参数
# 步骤（1-3与上述相同）：
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


#%% Xception预训练模型
# Xception默认图片大小为299*299*3
tf.keras.applications.xception.Xception(
        include_top=True, #是否包含顶层全连接层
        weigths='imagenet', #加载imagenet数据集上预训练的权重
        input_tensor=None,
        input_shape=None, #仅当include_top=False时有效，可输入自定义大小的图片，比如256*256*3
        pooling=None, #avg or max => 输出为(None, dim), 而None => 输出为(None,length,width,channel)
        classes=1000)

# 1）在Xception预训练模型上添加自定义层，进行训练
covn_base = tf.keras.applications.xception.Xception(include_top=False,
                                                    weigths='imagenet',
                                                    input_shape=(256,256,3),
                                                    pooling='avg')
covn_base.trainable = False
covn_base.summary()

model = keras.Sequential()
model.add(covn_base)
#model.add(layers.GlobalAveragePooling2D()) #由于Xception已经使用了pooling='avg'
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer=keras.optimizers.Adam(lr=0.0005),
              loss='binary_corssentropy',
              metrics=['acc'])
initial_epochs = 5
history = model.fit(train_image_ds,
                    steps_per_epoch=train_count//BATCH_SIZE,
                    epochs=initial_epochs,
                    validation_data=test_image_ds,
                    validation_steps=test_count//BATCH_SIZE)

# 2）解冻Xception的部分参数，结合新增自定义层进行fine-tune训练
covn_base.trainable = True
len(covn_base.layers) #预训练模型一共133层

fine_tune_at = -33
for layer in covn_base.layers[:fine_tune_at]:
    layer.trainable = False #除去后33层，其余都是不可训练的

model.compile(optimizer=keras.optimizers.Adam(lr=0.0005/10), #需要使用更小的lr
              loss='binary_corssentropy',
              metrics=['acc'])

fine_tune_epochs = 5
total_epochs = initial_epochs + fine_tune_epochs
history = model.fit(train_image_ds,
                    steps_per_epoch=train_count//BATCH_SIZE,
                    epochs=total_epochs,
                    initial_epoch = initial_epochs, #新增参数
                    validation_data=test_image_ds,
                    validation_steps=test_count//BATCH_SIZE)

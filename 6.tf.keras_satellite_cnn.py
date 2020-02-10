#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:30:12 2020

tf.keras卫星图像分类（二分类）

@author: tinghai
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0

#%% 卫星图像二分类示例(tf.data和CNN)
import pathlib
import random

# 获取图片路径
data_dir = './dateset/2_class'
data_root = pathlib.Path(data_dir)
for item in data_root.iterdir():
    print(item)

all_image_path = list(data_root.glob('*/*')) #获取固定正则表达式下的文件路径
all_image_path = [str(x) for x in all_image_path]
random.shuffle(all_image_path)
image_count = len(all_image_path) #1400

# 获取样本标签
label_names = sorted(item.name for item in data_root.glob('*/')) # ['airplane', 'lake']
label_to_index = dict((name, index) for index, name in enumerate(label_names)) # {'airplane':0, 'lake':1}

# pathlib.Path('xx/xx/xxxx.jpg').parent.name # 'lake'
all_image_label = [label_to_index[pathlib.Path(p).parent.name] for p in all_image_path] #获取每个样本的标签: 0/1

# 随机显示图像
import Ipython.display as display
index_to_label = dict(v,k for k,v in label_to_index.items) # {0:'airplane', 1:'lake'}
for n in range(3):
    image_index = random.choice(range(len(all_image_path)))
    display.display(display.Image(all_image_path[image_index]))
    print(index_to_label[all_image_label[image_index]]) # airplane

# 使用tensorflow读取图片
def load_preprosess_image(img_path):
#    img_path = all_image_path[0]
    img_raw = tf.io.read_file(img_path) #tf读取图片
#    img_tensor = tf.image.decode_image(img_raw, channels=3) #图片解码（即转换成的tensor数值矩阵），可以解析多种格式图片，但不能返回解析后的shape
    img_tensor = tf.image.decode_jpeg(img_raw, channels=3) #针对jpeg格式图像解码
#    img_tensor = tf.image.rgb_to_grayscale(img_tensor) #转换成单通道，即input_shape=(256,256,1)
#    img_tensor = tf.image.resize_image_with_crop_or_pad(img_tensor) #图像resize后不变形
    img_tensor = tf.image.resize(img_tensor, (256,256)) #图像可能发生变形，使用resize可以使得解析后的tensor具备shape
    print(img_tensor.shape) #[256,256,3]
    print(img_tensor.dtype) #tf.uint8
    img_tensor = tf.cast(img_tensor, tf.float32) #转换 tf.uint8 为 tf.float32
    img_tensor = img_tensor/255.0 #标准化
#    img_tensor = tf.image.per_image_standardization(img_tensor) #与上述类似的标准化
    img_numpy = img_tensor.numpy() #tensor转换成numpy
    print(img_numpy.max(), img_numpy.min())
    return img_tensor

# 针对解码的tensor，生成图片
plt.imshow(load_preprosess_image(all_image_path[100]))

# 使用tf.data构造训练和测试集
path_ds = tf.data.Dataset.from_tensor_slices(all_image_path)
image_dataset = path_ds.map(load_preprosess_image)
label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
print(image_dataset.shape)
print(label_dataset.shape)

for label in label_dataset.take(10):
    print(label.numpy())

for img in image_dataset.take(2):
    plt.imshow(img)

dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
test_count = int(image_count * 0.2)
train_count = image_count - test_count

train_dataset = dataset.skip(test_count)
test_dataset = dataset.take(test_count)

BATCH_SIZE = 32
train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# 构建模型
#   conv1-relu-bn-pooling-drop
#  -conv2-relu-bn-pooling-drop-...
#  -convn-relu-bn-globalpooling-dense-relu-bn-dense-relu-bn-sigmoid）
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape=(256,256,3), activation='relu')) # Relu和BN的位置可以互换
model.add(tf.keras.layers.Batchnormalization())
#model.add(tf.keras.layers.Activation('relu')) #也可单独添加激活函数
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
model.add(tf.keras.layers.Batchnormalization())
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu'))
model.add(tf.keras.layers.Batchnormalization())
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu'))
model.add(tf.keras.layers.Batchnormalization())
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Conv2D(1024, (3,3), activation='relu'))
model.add(tf.keras.layers.Batchnormalization())

model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Batchnormalization())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Batchnormalization())
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # 二分类

model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc']) # 使用二元交叉熵
step_per_epoch = train_count//BATCH_SIZE
validation_steps = test_count//BATCH_SIZE
history = model.fit(train_dataset, epochs=10, step_per_epoch=step_per_epoch, 
                    validation_data=test_dataset, validation_steps=validation_steps)

# 准确率评估
print(history.history.keys()) # losss, acc, val_loss, val_acc
plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()

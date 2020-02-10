#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:22:57 2020

tf.keras构建多输入模型

@author: tinghai
"""

#%% 多输出模型
import tensorflow as tf
from tensorflow import keras    
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np 
import glob
import os
import pathlib
import random

# 获取图片路径
data_dir = './dateset/moc'
data_root = pathlib.Path(data_dir)
for item in data_root.iterdir():
    print(item)

all_image_path = list(data_root.glob('*/*')) #获取给定路径下的所有文件路径
all_image_path = [str(x) for x in all_image_path]
random.shuffle(all_image_path)
image_count = len(all_image_path) #2525

# 获取样本标签
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir()) #获取给定路径下的所有一级文件夹名称,eg 'red_jeans'
color_label_names = set(name.split('_')[0] for name in label_names) #3 colors
item_label_names = set(name.split('_')[1] for name in label_names) #4 items

color_label_to_index = dict((name, index) for index, name in enumerate(color_label_names)) # {'black':0, 'red':1, 'blue':2}
item_label_to_index = dict((name, index) for index, name in enumerate(item_label_names)) # 

all_image_labels = [pathlib.Path(p).parent.name for p in all_image_path] #获取每个样本的标签: 0/1
color_labels = [color_label_to_index[p.split('_')[0]] for p in all_image_labels]
item_labels = [item_label_to_index[p.split('_')[1]] for p in all_image_labels]

color_index_to_label = dict(v,k for k,v in color_label_to_index.items)
item_index_to_label = dict(v,k for k,v in item_label_to_index.items)

# 随机取出图像查看
import Ipython.display as display
for n in range(3):
    image_index = random.choice(range(len(all_image_path)))
    display.display(display.Image(all_image_path[image_index]))
    print(all_image_label[image_index])

# plt.imshow()：针对解码的tensor，显示图片
# display.display(display.Image(image_path))：针对给定图片路径，显示图片

# 使用tensorflow读取图片
def load_preprosess_image(img_path):
    img_raw = tf.io.read_file(img_path) #tf读取图片
    img_tensor = tf.image.decode_jpeg(img_raw, channels=3) #针对jpeg格式图像解码
    img_tensor = tf.image.resize(img_tensor, (224,224)) #图像可能发生变形，使用resize可以使得解析后的tensor具备shape
    print(img_tensor.shape) #[256,256,3]
    print(img_tensor.dtype) #tf.uint8
    img_tensor = tf.cast(img_tensor, tf.float32) #转换 tf.uint8 为 tf.float32
    img_tensor = img_tensor/255.0 #标准化到[0,1]之间
    img_tensor = 2*img_tensor-1 #归一化到[-1,1]之间
    img_numpy = img_tensor.numpy() #tensor转换成numpy
    print(img_numpy.max(), img_numpy.min())
    return img_tensor

# 针对解码的tensor，生成图片
plt.imshow((load_preprosess_image(all_image_path[100])+1)/2) #恢复图片取值范围为[0,1]，传给imshow显示
plt.xlabel(all_image_labels[100])

# 生成image-dataset和label-dataset
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE #根据CPU数目，自动使用并行运算

path_ds = tf.data.Dataset.from_tensor_slices(all_image_path)
image_ds = path_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices((color_labels,item_labels))
print(image_ds.shape)
print(label_ds.shape)

for label in label_ds.take(2):
    print(label[0].numpy(), label[1].numpy())

for img in image_ds.take(2):
    plt.imshow(img)

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

# 划分训练集和测试集
test_count = int(image_count * 0.2)
train_count = image_count - test_count

train_data = image_label_ds.skip(test_count)
test_data = image_label_ds.take(test_count)

train_count = len(train_data)
train_data = train_data.shuffle(train_count).batch(BATCH_SIZE)
train_data = train_data.prefetch(buffer_size=AUTOTUNE) #在前台已读取数据的训练同时，预先读取后台数据
test_data = test_data.batch(BATCH_SIZE)

# model construction
mobile_net = tf.keras.applications.MobileNetV2(include_top=False,
                                               weigths=None, #仅使用MobileNetV2的架构，没有使用权重
                                               input_shape=(224,224,3))
inputs = tf.keras.Input(shape=(224,224,3))
x = mobile_net(inputs)
print(x.get_shape) #(None,7,7,1280)
x = tf.keras.layers.GlobalAveragePooling2D()(x) #or x = tf.keras.layers.Flatten()(x)
print(x.get_shape) #(None,1280)
x1 = tf.keras.layers.Dense(1024, activation='relu')(x)
x2 = tf.keras.layers.Dense(1024, activation='relu')(x)
out_color = tf.keras.layers.Dense(len(color_label_names), 
                                  activation='softmax',
                                  name='out_color')(x1)
out_item = tf.keras.layers.Dense(len(item_label_names), 
                                 activation='softmax',
                                 name='out_item')(x2)
model = tf.keras.Model(inputs=inputs, outputs=[out_color,out_item]) #单输入、多输出
model.summary()

# model training
model.compile(optimizer=keras.optimizers.Adam(lr=0.0005),
              loss={'out_color':'sparse_categorical_corssentropy', 'out_item':'sparse_categorical_corssentropy'}, 
              metrics=['acc'])

train_steps = train_count//BATCH_SIZE
test_steps = test_count//BATCH_SIZE
history = model.fit(train_data,
                    steps_per_epoch = train_steps,
                    epochs = 15,
                    batch_size = BATCH_SIZE,
                    validation_data = test_data,                    
                    validation_steps = test_steps)

# model evaluation
model.evaluate(test_image_array, [test_color_labels, test_item_labels], verbose=0)

# model predict
my_image = load_preprosess_image(r'{}'.format(random.choice(test_dir)))
#my_image = load_preprosess_image(all_image_path[0])
pred = model.predict(np.expend_dims(my_image, axis=0)) #需扩展第一维的Batch_size => (None,224,224,3)
# or
# pred = model(np.expend_dims(my_image, axis=0), training=False) #直接使用model()调用的方式

pred_color = color_index_to_label.get(np.argmax(pred[0][0])) #预测概率最大的颜色
pred_item = item_index_to_label(np.argmax(pred[1][0])) #预测概率最大的商品
plt.imshow((load_preprosess_image(my_image)+1)/2)
plt.xlabel(pred_color + '_' + pred_item)


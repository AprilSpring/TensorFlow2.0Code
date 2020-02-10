#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 21:28:05 2020

图像语义分割 - FCN model

@author: tinghai
"""

#%% 图像语义分割
# 目标：预测图像中每个像素的类别（区分背景、边缘、同类实体）
# Model-1：FCN (Fully convolutional network)，相比较于分类网络，FCN最后不使用全连接层，而使用上采样和跳接结构，还原至原图像大小
# Model-2：Unet，从更少的图像中进行学习，


#%% 图像语义分割 - FCN（同图像定位的数据）
# 输入：任意尺寸彩色图像
# 输出：与输入尺寸相同
# 通道数：n(目标类别数 + 1(背景)

# 1) Upsampling：插值法、反池化、反卷积（转制卷积）
# 反卷积：
    # 通过训练来放大图片
    # tf.keras.layers.Conv2DTranspose(filters=32,kernal_size=(3,3),stries=(1,1),padding='same')
# 2) 跳接结构
    # 用于结合前面的局部特征和后面的全局特征

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
%matplotlib inline
from lxml import etree
import numpy as np
import glob

#%% 1）原图像与语义分割图像处理
images = glob.glob('./images/*.jpg') 
anno = glob.glob('./annotations/trimaps/*.png')

# shuffle
np.random.seed(2)
index = np.random.permutation(len(images))
images = np.array(images)[index]
anno = np.array(anno)[index]
dataset = tf.data.Dataset.from_tensor_slices((images, anno))

# 原图像解码
def read_jpg(path):
#    img = tf.io.read_file('./images/yorkshire_terrier_99.jpg')
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
#    plt.imshow(img)
    return img

# 语义图像解码
def read_png(path):
#    img = tf.io.read_file('./images/yorkshire_terrier_99.jpg')
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
#    print(img.shape) #(358,500,1)
#    img = tf.squeeze(img)
#    print(img.shape) #(358,500)
#    plt.imshow(img)
#    print(np.unique(img.numpy())) #array([1,2,3])
    return img

# 图片大小
def resize_img(images):
    images = tf.image.resize(images, [224,224])
    return images

# 归一化
def normal_img(input_images, input_anno):
    input_images = tf.cast(input_images, tf.float32)
    input_images = input_images/127.5 - 1 #[-1,1]
    input_anno = input_anno -1 #[0,1,2] 
    return input_images, input_anno

def load_images(input_images_path, input_anno_path):
    input_images = read_jpg(input_images_path)
    input_anno = read_png(input_anno_path)
    input_images = resize_img(input_images)
    input_images = resize_img(input_anno)
    return normal_img(input_images, input_anno)

BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE #根据CPU数目，自动使用并行运算
dataset = dataset.map(load_images, num_parallel_calls=AUTOTUNE)

test_count = len(images) * 0.2
train_count = len(images) - test_count

data_train = dataset.skip(test_count)
data_train = data_train.shuffle(100).repeat().batch(BATCH_SIZE)
data_train = data_train.prefetch(AUTOTUNE)
print(data_train.shape) #((None,224,224,3), (None,224,224,1))

data_test= dataset.take(test_count)
data_test = data_test.batch(BATCH_SIZE)
data_test = data_test.prefetch(AUTOTUNE) 

# 同时显示原图像和语义分割图像
for img, anno in data_train.take(1):
    plt.subplot(2,1,1)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
    plt.subplot(2,1,2)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(anno[0]))

#%% 2）预训练模型构建
conv_base = tf.keras.applications.VGG16(include_top=False,
                                        weigths='imagenet',
                                        input_shape=(256,256,3))
conv_base.summary()
# 最后一层(7，7，512)=>上采样为(14，14，512)=>与上层输出相加(14，14，512)=>再上采样为(28，28，256)=>与上层输出相加(28，28，256)=> ...=> 最终输出(224,224,1)

#%% 3）获得模型中间层的输出
# 如何获取网络中某层的输出？
conv_base.get_layer('block5_conv3').output

# 如何获取子模型？ (子模型继承了原模型的权重)
sub_model = tf.keras.models.Model(inputs=conv_base.input, output=conv_base.get_layer('block5_conv3').output)
sub_model.summary()

# 获取预训练模型的多个中间层的输出
layer_names = ['block5_conv3', 'block4_conv3', 'block3_conv3', 'block5_pool'] #要获取的中间层名称
layers_output = [conv_base.get_layer(layer_name).output for layer_name in layer_names]
multi_out_model = tf.keras.models.Model(inputs=conv_base.input, output=layers_output)
print(multi_out_model.predict(image)) #分别返回4个子模型的预测输出结果
multi_out_model.trainable = False

#%% 4）FCN模型构建
inputs = tf.keras.layers.Input(shape=(224,224,3))
out_block5_conv3, out_block4_conv3, out_block3_conv3, out = multi_out_model(inputs)
print(out.shape) #(None,7,7,512)
print(out_block5_conv3.shape) #(None,14,14,512)
print(out_block4_conv3.shape) #(None,28,28,512)
print(out_block3_conv3.shape) #(None,56,56,256)

# a) 针对out进行upsampling
x1 = tf.keras.layers.Conv2DTranspose(filters=512,
                                     kernal_size=(3,3),
                                     stries=2, #变为原来的2倍大小
                                     padding='same',
                                     activation='relu')(out) #(None,14,14,512)
# 增加一层卷积，增加特征提取程度
x1 = tf.keras.layers.Conv2D(filters=512,
                            kernal_size=(3,3),
                            padding='same',
                            activation='relu')(x1) #(None,14,14,512)
# 与上层进行相加
x2 = tf.add(x1, out_block5_conv3) #(None,14,14,512)

# b) 针对x2进行upsampling
x2 = tf.keras.layers.Conv2DTranspose(filters=512,
                                     kernal_size=(3,3),
                                     stries=2, #变为原来的2倍大小
                                     padding='same',
                                     activation='relu')(x2) #(None,28,28,512)
# 增加一层卷积，增加特征提取程度
x2 = tf.keras.layers.Conv2D(filters=512,
                            kernal_size=(3,3),
                            padding='same',
                            activation='relu')(x2) #(None,28,28,512)
# 与上层进行相加
x3 = tf.add(x2, out_block4_conv3) #(None,28,28,512)

# c) 针对x2进行upsampling
x3 = tf.keras.layers.Conv2DTranspose(filters=256,
                                     kernal_size=(3,3),
                                     stries=2, #变为原来的2倍大小
                                     padding='same',
                                     activation='relu')(x3) #(None,56,56,256)
# 增加一层卷积，增加特征提取程度
x3 = tf.keras.layers.Conv2D(filters=256,
                            kernal_size=(3,3),
                            padding='same',
                            activation='relu')(x3) #(None,56,56,256)
# 与上层进行相加
x4 = tf.add(x3, out_block3_conv3) #(None,56,56,256)

# d) 针对x4进行upsampling
x5 = tf.keras.layers.Conv2DTranspose(filters=64,
                                     kernal_size=(3,3),
                                     stries=2, #变为原来的2倍大小
                                     padding='same',
                                     activation='relu')(x4) #(None,112,112,64)
# 增加一层卷积，增加特征提取程度
x5 = tf.keras.layers.Conv2D(filters=64,
                            kernal_size=(3,3),
                            padding='same',
                            activation='relu')(x5) #(None,112,112,64)
# d) 针对x5进行upsampling
prediction = tf.keras.layers.Conv2DTranspose(filters=3,
                                     kernal_size=(3,3),
                                     stries=2, #变为原来的2倍大小
                                     padding='same',
                                     activation='softmax')(x5) #(None,224,224,3)
# 最终创建模型
model = tf.keras.models.Model(inputs=inputs, outputs=prediction)
model.summary()

#%% 5）FCN模型训练
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['acc'])
history = model.fit(data_train, 
                    epochs=5, 
                    steps_per_epoch=train_count//BATCH_SIZE,
                    validation_data=data_test, 
                    validation_steps=test_count//BATCH_SIZE)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(EPOCHS)
plt.figure()
plt.plot(epochs, loss, 'r', label='train-loss')
plt.plot(epochs, val_loss, 'b', label='test-loss')
plt.title('Train and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim([0,1])
plt.legend()
plt.show()

#%% 6) 模型预测
for img, mask in data_test.take(1): #take one batch
    pred = model.predict(img)
    pred = tf.argmax(pred, axis=-1)
    pred = pred[..., tf.newaxis] #扩展维度 (None,224,224,1)
    
    plt.figure(figsize=(10,10))
    num = 5
    for i in range(num):
        plt.subplot(num, 3, i*3+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(img[i])) #原图
        plt.subplot(num, 3, i*3+2)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(mask[i])) #语义分割图
        plt.subplot(num, 3, i*3+3)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred[i])) #预测的语义分割图
        

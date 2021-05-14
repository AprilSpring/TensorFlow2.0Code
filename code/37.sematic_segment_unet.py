#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Created on: 2020/5/11 11:30 下午

@annotaion: 图像语义分割的unet模型（城市街景数据集-cityscapes）
            适用于从小型数据集得到一个性能较好的语义分割模型，比如医疗图像识别
            unet结构：前半部分是特征提取（即编码器）、后半部分是上采样（即解码器）
@author: tinghai
"""

# unet上采样使用的是tf.concat，而FCN上采样使用的tf.add，注意区分

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

##% 数据读入
img = glob.glob('./dataset/leftImage8bit/train/*/*.png')
label = glob.glob('./dataset/gtFine/train/*/*_gtFine_labelIds.png')
len(img, label)

img_val = glob.glob('./dataset/leftImage8bit/test/*/*.png')
label_val = glob.glob('./dataset/gtFine/test/*/*_gtFine_labelIds.png')
len(img_val, label_val)

# 训练集乱序
index = np.random.permutation(len(img))
img = np.array(img)[index]
label = np.array(label)[index]

dataset_train = tf.data.Dataset.from_tensor_slices((img, label))
dataset_val = tf.data.Dataset.from_tensor_slices((img_val, label_val))


def read_png(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    return img

def read_png_label(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    return img

img_1 = read_png(img[0])
label_1 = read_png(label[0])
print(img_1, label_1) #(1024,2048,3), (1024,2048,1)


##% 数据增强
# 1）引入随机翻转
if tf.random.uniform(()) > 0.5:
    img = tf.image.flip_left_right(img) #保证img和label中的图片对应翻转
    label = tf.image.flip_left_right(label)

def crop_img(img, label):
    # 2）随机裁剪:保证img和label的图像裁剪是一致的，因此将img和label进行叠加
    concat_img = tf.concat([img, label], axis=-1)  # channel维度合并
    # print(concat_img)  # (1024,2048,4)

    # 3) 图像过大，首先需要缩小一些，再进行裁剪
    concat_img = tf.image.resize(concat_img, (280, 280),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    crop_img = tf.image.random_crop(concat_img, [256,256,4])
    return crop_img[:,:,:3], crop_img[:,:,3:]

img_1, label_1 = crop_img(img_1, label_1)
print(img_1, label_1) #(256,256,3),(256,256,1)

plt.subplot(1,2,1)
plt.imshow(img_1.numpy()) #显示图像
plt.subplot(1,2,2)
plt.imshow(np.squeeze(label_1.numpy())) #显示标签, imshow只能接收channel为3，或者2维图片

# 4) 标准化
def normal(img, label):
    img = tf.cast(img, tf.float32)/127.5 -1 #[-1,1]
    label = tf.cast(label, tf.int32)
    return img, label

# 5) 数据增强
def load_image_train(img_path, label_path):
    img = read_png(img_path)
    label = read_png_label(label_path)
    img, label = crop_img(img, label)

    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)  # 保证img和label中的图片对应翻转
        label = tf.image.flip_left_right(label)

    img, label = normal(img, label)
    return img, label

def load_image_val(img_path, label_path):
    img = read_png(img_path)
    label = read_png_label(label_path)
    # 不需要进行数据增强，直接进行resize()
    img = tf.image.resize(img, (256, 256)
    label = tf.image.resize(label, (256, 256)
    img, label = normal(img, label)
    return img, label

BATCH_SIZE = 32
SUFFER_SIZE = 300 #前面已经乱序，此处可以用一个较小的值
train_count = len(label)
val_count = len(label_val)
step_per_epoch = train_count//BATCH_SIZE
val_step = val_count//BATCH_SIZE
auto = tf.data.experimental.AUTOTUNE

dataset_train = dataset_train.map(load_image_train, num_parallel_calls=auto)
dataset_val = dataset_val.map(load_image_val, num_parallel_calls=auto)

# 绘图
for i, m in dataset_train.take(1):
    plt.subplot(1, 2, 1)
    plt.imshow((i.numpy() + 1) / 2)  # 显示图像
    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(m.numpy()))  # 显示标签, imshow只能接收channel为3，或者2维图片

dataset_train = dataset_train.cache().repeat().shuffle(SUFFER_SIZE).batch(BATCH_SIZE).prefetch(auto)
dataset_val = dataset_train.cache().batch(BATCH_SIZE)


##% 模型
np.unique(label_1.numpy()) #查看第一张图片的label的唯一值，一共34类

def create_model():
    inputs = tf.keras.layers.Input(shape=(256,256,3))

    # 1）卷积和下采样部分
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x) #(256,256,64)

    x1 = tf.keras.layers.MaxPool2D(padding='same')(x) #下采样，(128,128,64)

    x1 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)  # (128,128,128)
    x1 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)  # (128,128,128)

    x2 = tf.keras.layers.MaxPool2D(padding='same')(x1)  # 下采样，(64,64,128)

    x2 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)  # (64,64,256)
    x2 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)  # (64,64,256)

    x3 = tf.keras.layers.MaxPool2D(padding='same')(x2)  # 下采样，(32,32,256)

    x3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)  # (32,32,512)
    x3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)  # (32,32,512)

    x4 = tf.keras.layers.MaxPool2D(padding='same')(x3)  # 下采样，(16,16,512)

    x4 = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu')(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)  # (16,16,1024)
    x4 = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu')(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)  # (16,16,1024)

    # 2）上采样部分
    # 上采样-1
    x5 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2, padding='same', activation='relu')(x4) #512个大小为2的卷积核
    x5 = tf.keras.layers.BatchNormalization()(x5) # (32,32,512)，与x3大小一致

    x6 = tf.concat([x3,x5], axis=-1) #最后一个维度进行合并，(32,32,1024)

    x6 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x6)
    x6 = tf.keras.layers.BatchNormalization()(x6)  # (32,32,512)
    x6 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x6)
    x6 = tf.keras.layers.BatchNormalization()(x6)  # (32,32,512)

    # 上采样-2
    x7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding='same', activation='relu')(x6)
    x7 = tf.keras.layers.BatchNormalization()(x7)  # (64,64,256)，与x2大小一致

    x8 = tf.concat([x2, x7], axis=-1)  # 最后一个维度进行合并，(64,64,512)

    x8 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x8)
    x8 = tf.keras.layers.BatchNormalization()(x8)  # (64,64,256)
    x8 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x8)
    x8 = tf.keras.layers.BatchNormalization()(x8)  # (64,64,256)

    # 上采样-3
    x9 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same', activation='relu')(x8)
    x9 = tf.keras.layers.BatchNormalization()(x9)  # (128,128,128)，与x1大小一致

    x10 = tf.concat([x1, x9], axis=-1)  # 最后一个维度进行合并，(128,128,256)

    x10 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x10)
    x10 = tf.keras.layers.BatchNormalization()(x10)  # (128,128,128)
    x10 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x10)
    x10 = tf.keras.layers.BatchNormalization()(x10)  # (128,128,128)

    # 上采样-4
    x11 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same', activation='relu')(x10)
    x11 = tf.keras.layers.BatchNormalization()(x11)  # (256,256,64)，与x大小一致

    x12 = tf.concat([x, x11], axis=-1)  # 最后一个维度进行合并，(256,256,128)

    x12 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x12)
    x12 = tf.keras.layers.BatchNormalization()(x12)  # (256,256,64)
    x12 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x12)
    x12 = tf.keras.layers.BatchNormalization()(x12)  # (256,256,64)

    # 3) 输出
    output = tf.keras.layers.Conv2D(34, 1, padding='same', activation='softmax')(x12) #34分类，(256,256,34)
    return tf.keras.Model(inputs=inputs, outputs=output)


model = create_model()
model.summary()
tf.keras.utils.plot_model(model) #need to 'conda install pydot'
# tf.keras.metrics.MeanIoU(num_classes=34) #根据one-hot编码进行计算的，因此修改class MeanIoU

class MeanIoU(tf.keras.metrics.MeanIoU):
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc', MeanIoU(num_classes=34)])

EPOCHS = 60
history = model.fit(dataset_train,
                    epochs=EPOCHS,
                    step_per_epoch=step_per_epoch,
                    validation_steps=val_step)

num = 3
for image, mask in dataset_val.take(1):
    pred_mask = model.predict(image)
    pred_mask = tf.argmax(pred_mask, axis=1)
    pred_mask = pred_mask[..., tf.newaxis]

    plt.figure(figsize=(10,10))
    for i in range(num):
        plt.subplot(num, 3, i*num + 1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(image[i]))
        plt.subplot(num, 3, i * num + 2)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(mask[i]))
        plt.subplot(num, 3, i * num + 3)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[i]))



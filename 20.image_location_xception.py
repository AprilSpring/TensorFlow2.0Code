#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:03:23 2020

图像定位预测

@author: tinghai
"""

#%% 图像算法
# 1）图像分类
# 2）图像分类+定位（矩形框标识）
# 3）语义分割（semantic segmentation, 区分图像中每个像素点的类别，比如人和狗）
# 4）目标检测（object localization, 矩形框标识别所有物体的位置）
# 5）实例分割（类似语义分割，但对每个实体都进行区分，比如不同只狗）


#%% 图像定位（有监督的回归问题，使用L1/L2损失）
# Oxford-IIIT pet dataset：包含37种宠物，每种宠物200张图片左右，及每种宠物的类别、头部轮廓标注、语义分割信息。
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
%matplotlib inline
from lxml import etree
import numpy as np
import glob

tf.test.is_gpu_available()

#%% 1）读取图像文件和位置文件
images = glob.glob('./images/*.jpg') 
xlms = glob.glob('./annotations/xlms/*.xlm')
print(len(images)) #6000+
print(len(xlms)) #3686

names = [x.split('/')[-1].split('.')[0] for x in xlms]
imgs_train = [x for x in images if x.split('/')[-1].split('.')[0] in names] #标记位置的图片
imgs_test = [x for x in images if x.split('/')[-1].split('.')[0] not in names] #未标记位置的图片
imgs_train.sort(key=lambda x: x.split('/')[-1].split('.')[0])
xlms.sort(key=lambda x: x.split('/')[-1].split('.')[0])
print(imgs_train[-5:])
print(xlms[-5:])

#%% 2）获取label_datasets
# xml文件的位置信息提取
def to_labels(path):
#    xml = open('./annotations/Abyssinian_1.xml').read()
    xml = open(path).read()
    sel = etree.HTML(xml)
    width = int(sel.xpath('//size/width/text()')[0])
    height = int(sel.xpath('//size/height/text()')[0])
    xmin = int(sel.xpath('//bndbox/xmin/text()')[0])
    xmax = int(sel.xpath('//bndbox/xmax/text()')[0])
    ymin = int(sel.xpath('//bndbox/ymin/text()')[0])
    ymax = int(sel.xpath('//bndbox/ymax/text()')[0])
    # 针对缩放的图片，对位置信息进行相应比例的转换
#    xmin = (xmin/width)*224
#    xmax = (xmax/width)*224
#    ymin = (ymin/height)*224
#    ymax = (ymax/height)*224
    return [xmin/width, xmax/width, ymin/height, ymax/height]

labels = [to_labels(path) for path in xlms]
out1, out2, out3, out4 = list(zip(*labels))
out1 = np.array(out1)
out2 = np.array(out2)
out3 = np.array(out3)
out4 = np.array(out4)
label_datasets = tf.data.Dataset.from_tensor_slice((out1,out2,out3,out4))

#%% 3）获取image_datasets
# 图片解码
def read_ipg(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

# 由于图片大小不同，因此需要对图片进行缩放
def normalize(input_image):
    img = tf.image.resize(input_image, [224,224])
    img = img/127.5 -1 #归一化到[-1,1]

@tf.function
def load_image(path):
    img = read_ipg(path)
#    print(img.shape) #(400,600,3)
#    plt.imshow(img)
    img = normalize(img)
#    plt.imshow(img)
    return img

image_paths = tf.data.Dataset.from_tensor_slice(imgs_train)
image_datasets = image_paths.map(load_image)

BATCH_SIZE = 32
datasets = tf.data.Dataset.zip((image_datasets,label_datasets))
image_count = len(image_count)
test_count = int(image_count * 0.2)
train_count = image_count - test_count
train_datasets = datasets.skip(test_count)
test_datasets = datasets.take(test_count)
train_datasets = train_datasets.shuffle(train_count).repeat().batch(BATCH_SIZE)
test_datasets = test_datasets.batch(BATCH_SIZE)

#%% 4）恢复datasets的图片及位置框
for img, label in train_datasets.take(1): #take one batch
    plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
    out1,out2,out3,out4 = label
    xmin,ymin,xmax,ymax = out1[0].numpy()*224, out2[0].numpy()*224, out3[0].numpy()*224, out4[0].numpy()*224
    # 图片上的矩形框绘制
    rect = Rectangle((xmin, ymin),(xmax-xmin),(ymax-ymin),fill=False, color='red') #起始点坐标、x轴长度、y轴长度
    ax = plt.gca() #获取当前图像 get_current_image
    ax.axes.add_patch(rect) #当前图像中添加矩形框
    plt.show()

#%% 5）构建图像定位的预测模型（回归问题）
xception = tf.keras.applications.xception.Xception(include_top=False,
                                                    weigths='imagenet',
                                                    input_shape=(224,224,3),
                                                    pooling='avg')
xception.trainable = False
xception.summary()

inputs = tf.keras.layers.Input(shape=(224,224,3))
x = xception(inputs)
x = tf.keras.layers.GlobalAveragePooling2D(x)
x = tf.keras.layers.Dense(4096,activation='relu')(x)
x = tf.keras.layers.Dense(1000,activation='relu')(x)
y1 = tf.keras.layers.Dense(1)(x)
y2 = tf.keras.layers.Dense(1)(x)
y3 = tf.keras.layers.Dense(1)(x)
y4 = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=inputs, outputs=[y1, y2, y3, y4]) #单输入、多输出
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), #lr需要较小
              loss='mse',
              metrics=['mae'])
EPOCHS = 10
history = model.fit(train_datasets,
                    steps_per_epoch=train_count//BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=test_datasets,
                    validation_steps=test_count//BATCH_SIZE)

#%% 6）绘制预测效果
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(EPOCHS)
plt.figure()
plt.plot(epochs, loss, 'r', label='train-loss')
plt.plot(epochs, val_loss, 'b', label='test-loss')
plt.title('Train and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%% 7）模型保存与加载
model.save('./image_location_detect_v1.h5')
new_model = tf.keras.models.load_model('./image_location_detect_v1.h5')

#%% 8）预测结果验证
plt.figure(figsize=(8,24))
for img, _ in test_datasets.take(1): #take one batch
    out1,out2,out3,out4 = new_model.predict(img)
    # 显示3张图片，及预测的矩形框
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(img[i]))
        xmin,ymin,xmax,ymax = out1[i].numpy()*224, out2[i].numpy()*224, out3[i].numpy()*224, out4[i].numpy()*224
        # 图片上的矩形框绘制
        rect = Rectangle((xmin, ymin),(xmax-xmin),(ymax-ymin),fill=False, color='red') #起始点坐标、x轴长度、y轴长度
        ax = plt.gca() #获取当前图像 get_current_image
        ax.axes.add_patch(rect) #当前图像中添加矩形框

#%% 9）图像定位的评价指标
# IoU：Intersection over Union(交并比)，即预测边框和真实边框的交集和并集的比值。

#%% 10）优化方向
# 先大后小：先预测出关键点，后在关键点周边预测范围
# 图片划窗：是否有关键点、关键点的位置
# 针对不确定实体个数的预测问题：先检测多个对象，然后在多个对象上回归出位置
# 变回归为分类问题：即定位区域的像素点为1，其余位置为0


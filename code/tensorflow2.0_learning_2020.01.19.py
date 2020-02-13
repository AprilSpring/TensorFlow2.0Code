#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:13:13 2020

TensorFlow2.0 常用函数

tf.keras：构建和训练模型（Sequential模式、函数式API）
Eager模式：直接迭代和直观调试
tf.GradientTape：求解梯度，自定义训练模式
tf.data：加载图片与结构化数据
tf.function：自动图运算

@author: liutingting16
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0


#%% tf.keras
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,))) # 第一个参数是输出维度
model.summary()
model.compile(optimizer='adam', loss='mse')
model.fit(train_x, train_y, epochs=30)
model.predict(text_x)


#%% 二分类：logistic model
train_x = np.random.rand(100,3)
train_y = np.random.randint(0,2,(100,1))
test_x = np.random.rand(100,3)
test_y = np.random.randint(0,2,(100,1))
model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'), #第一层需要定义输入数据的维度：input_shape
                             tf.keras.layers.Dense(5, activation='relu'),
                             tf.keras.layers.Dense(1, activation='sigmoid')])
model.summary()
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['acc'])
#history = model.fit(train_x, train_y, epochs=300)
history = model.fit(train_x, train_y, epochs=300, validation_data=(test_x, test_y)) # 在每个epoch上评估测试集准确率
model.predict(test_x)

print(history.history.keys()) # losss, acc, val_loss, val_acc
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()


#%% mnist多分类：softmax交叉熵
(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
plt.imshow(train_image[0])
train_image = train_image/255.0 # 0-1值
test_image = test_image/255.0 # 0-1值

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # 将张量拉平成同一维度 (28,28) -> 28*28
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()
model.compile(opitimizer=tf.keras.optimizer.Adam(lr=0.01),
              loss='sparse_categorical_corssentropy', # 当label为顺序数字编码时
#              loss='categorical_corssentropy', # 当label为onehot编码时
              metrics=['acc'])
history = model.fit(train_image, train_label, epochs=30)
model.evaluate(test_image, test_label) # 评估
predict = model.predict(test_image) #预测
print(predict[0]) # 返回预测概率
print(np.argmax(predict[0])) # 返回最大概率的位置
print(test_label[0]) # 真实标签

train_label_hoehot = tf.keras.utils.to_categorial(train_label) # 转化train_label为onehot标签
test_label_hoehot = tf.keras.utils.to_categorial(test_label) # 转化test_label为onehot标签


#%% 函数式API
input1 = tf.keras.Input(shape=(28,28))
input2 = tf.keras.Input(shape=(28,28))
x1 = tf.keras.layers.Flatten()(input1)
x2 = tf.keras.layers.Flatten()(input2)
x = tf.keras.layers.concatenate([x1,x2])
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs=[input1, input2], outputs=outputs) # 多输入，单输出
model.summary()
model.compile() # 方式同上


#%% tf.data
# 创建dataset的几种方式
dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5]) 

dataset = tf.data.Dataset.from_tensor_slices([[1,2],[3,4],[5,6]])

dataset = tf.data.Dataset.from_tensor_slices({'a':[1,2,3,4],
                                              'b':[6,7,8,9],
                                              'c':[12,13,14,15]})
dataset = tf.data.Dataset.from_tensor_slices(np.array([1,2,3,4,5])) 

for ele in dataset:
    print(ele)

for ele in dataset:
    print(ele.numpy()) # 转换回numpy数据格式

for ele in dataset.take(4): # 提取topN
    print(ele.numpy())

# shuffle, repeat, batch的使用
dataset = dataset.shuffle(buffer_size=5, seed=0) # 打乱
dataset = dataset.repeat(count=3) # 重复
dataset = dataset.batch(batch_size=3)
for ele in dataset:
    print(ele.numpy())

# 数据变换：map
dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5]) 
dataset = dataset.map(tf.square)
print([ele.numpy() for ele in dataset])

# mnist示例
(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
train_image = train_image/255.0 # 0-1值
test_image = test_image/255.0 # 0-1值
print(train_image.shape) # (60000, 28, 28)

ds_train_img = tf.data.Dataset.from_tensor_slices(train_image)
ds_train_lab = tf.data.Dataset.from_tensor_slices(train_label)
ds_train = tf.data.Dataset.zip((ds_train_img, ds_train_lab)) #两个tensor的对应位置元素合并，((28,28),())

ds_test = tf.data.Dataset.from_tensor_slices((test_image, test_label)) #同ds_train生成的效果一样，((28,28),())

ds_train = ds_train.shuffle(10000).repeat().batch(64)
ds_test = ds_test.batch(64) # 默认使用了repeat()

model = tf.keras.Sequential(tf.keras.layers.Flatten(input_shape=(28, 28)),
                            tf.keras.layers.Dense(128, activation='relu'),
                            tf.keras.layers.Dropout(0.1),
                            tf.keras.layers.Dense(10, activation='softmax'))
model.summary()
model.compile(opitimizer=tf.keras.optimizer.Adam(lr=0.01),
              loss='sparse_categorical_corssentropy', # 当label为顺序数字编码时
              metrics=['acc'])
history = model.fit(ds_train, 
                    epochs=5, 
                    steps_per_epoch=train_image.shape[0]//64, # "//" used to be int
                    validation_data=ds_test, 
                    validation_steps=test_image.shape[0]//64) 


#%% CNN
import keras
from keras import layers

layers.Conv2D(filters, #卷积核数量（即卷积后的通道数）
              kernal_size, #卷积核大小
              strides=(1,1), #步长为1
              padding='valid', # 'same'
              activation='relu',
              use_bias=True,
              kernel_initializer='glorot_uniform',
              bias_initializer=None,
              kernel_regularizer=None, #正则化
              bias_regularizer=None)
              
         
layers.MaxPooling2D(pool_size=(2,2),
                    strides=None,
                    padding='valid')

# mnist示例
# !pip install -q tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf
tf.test.is_gpu_available()

(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
train_image = np.expand_dims(train_image, -1) # 或reshape(), -1表示扩增的最后一个维度，生成[样本量，长，宽，通道]，与上述使用Flatten不同
test_image = np.expand_dims(test_image, -1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3), 
                                 input_shape=train_image.shape[1:], #shape[1:]表示除去第一维度，即去除batch的维度，首次需要定义该参数
                                 activation='relu',
                                 padding='same'))
print(model.output_shape) #(None, 28, 28 ,32)
model.add(tf.keras.layers.MaxPooling2D()) # default pooling_size=(2,2), #(None, 14, 14 ,32)
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Conv2D(64, (3,3), 
                                 activation='relu',
                                 padding='same')) #(None, 14, 14 ,64)
model.add(tf.keras.layers.MaxPooling2D()) #(None, 7, 7 ,64)
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.GlobalAveragePooling2D()) #全局平均池化，或使用Flatten()使其变成1个维度, (None, 64)
model.add(tf.keras.layers.Dense(128, activation='relu')) #FFN, (None, 128)
model.add(tf.keras.layers.Dense(10, activation='softmax')) #softmax层，(None, 10)
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
history = model.fit(train_image, train_label, epochs=10, validation_data=(test_image, test_label))
print(history.history.keys()) # losss, acc, val_loss, val_acc
plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()


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
model.add(tf.keras.layers.Batchnormalization()) #放在卷基层后
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


#%% 文本向量化
from tensorflow import keras
from tensorflow.keras import layers

# 电影评论数据
data = keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = data.load_data(num_words=10000) #已经将文本转化成ID
#d = data.get_word_index()
#print(np.mean([len(x) for x in x_train])) # 238
x_train = keras.preprocessing.sequence.pad_sequences(x_train, 300) #填充0,使得长度为300
x_test = keras.preprocessing.sequence.pad_sequences(x_test, 300)

#test = 'i am a student ahh'
#[d[x] if x in d.keys() else 0 for x in test.split()]
#{x:d[x] for x in test.split() if x in d.keys()}

# 构建模型
model = keras.models.Sequential()
model.add(layers.Embedding(10000, 50, input_length=300)) #向量化，input_length输入数据的长度, (None, 300, 50)
model.add(layers.Flatten()) #将输入展平，不影响批量大小，(None, 15000)
model.add(layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))) #添加L2正则化
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
              loss='binary_crossentropy',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=256, validation_data=(x_test, y_test))


#%% Eager模式（命令行式编写环境/tensroflow的交互模式）
print(tf.executing_eagerly()) #True

# tensor可以与numpy直接运算
a = tf.constant([[1,2],[3,4]])
b = tf.add(a, 1)
c = tf.multiply(a, b)

num = tf.convert_to_tensor(10)
for i in range(num.numpy()):
    i = tf.constant(i)
    if int(i % 2) == 0:
        print(i)

d = np.array([[5,6],[7,8]])
print(a + d)
print((a + d).numpy())

# 变量
v = tf.Variable(0.0)
print(v+1)

v.assign(5) #改变变量的值
v.assign_add(1) #变量值加1
v.read_value() #返回变量值

# 梯度运算
w = tf.Variable([[3.0]]) #需要是float数据类型
with tf.GradientTape() as t:
    loss = w*w + w
grad = t.gradient(loss, w) # 求解loss对w的微分

w = tf.constant([[3.0]])
with tf.GradientTape() as t:
    t.watch(w) # 针对常量w进行跟踪，以便于后续使用t.gradient()求导，Variable不需要watch()
    loss = w*w + w
grad2 = t.gradient(loss, w)

w = tf.constant([[3.0]])
with tf.GradientTape(persistent=True) as t: #persistent=True用于多次计算微分
    t.watch(w)
    y = w*w + w
    z = y*y
grad3 = t.gradient(y, w)
grad4 = t.gradient(z, w)


#%% eager自定义训练模式 - minst示例
'''
步骤：
    1)按Batch准备数据
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(10000).batch(32)
    2)定义模型结构
        model = tf.keras.Sequential([...])
    3)选择optimizer
        optimizer = tf.keras.optimizer.Adam()
    4)计算loss
        y_ = model(x)
        loss = loss_func(y,y_)
    5)计算grads
        with tf.GradientTape() as t
        grads = t.gradient(loss, model.trainable_variables)
    6)optimizer按照grads方向更新参数
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    7)按batch进行训练
        重复 5)和 6）
'''

# 生成数据集
(train_image, train_labels), (test_image, test_labels) = tf.keras.datasets.mnist.load_data()

# 训练集
train_image = tf.expand_dims(train_image, -1) #-1表示扩增的最后一个维度，由于使用CNN因此需要扩增数据维度
train_image = tf.cast(train_image/255, tf.float32) #需要float类型才能做梯度运算
train_labels = tf.cast(train_labels, tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels))
dataset = dataset.shuffle(10000).batch(32) # 默认repeat(1)；如果使用fit方法的话，需添加repeat(),无限循环

# 测试集
test_image = tf.expand_dims(test_image, -1)
test_image = tf.cast(test_image/255, tf.float32)
test_labels = tf.cast(test_labels, tf.int64)
test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_labels))
test_dataset = test_dataset.batch(32)

# 模型构建
model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16,[3,3], activation='relu',input_shape=(28,28,1)), #任意图片大小：input_shape=(None,None,1)
        tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
        tf.keras.layser.GlobalMaxPooling2D(), #GlobalAveragePooling2D()
        tf.keras.layers.Dense(10, activation='softmax')])

# 自定义模型优化（不使用compile）
optimizer = tf.keras.optimizer.Adam(lr=0.01) #初始化优化器
#loss_func = tf.keras.losses.sparse_categorial_crossentropy(y_true, y_pred, from_logits = False) #是否从上层Dense激活，如果是则True，否则False
# or
loss_func = tf.keras.losses.SparseCategorialCrossentropy(from_logits=False) #返回一个方法，loss_func(y, y_)

features, labels = next(iter(dataset)) #按照batch迭代返回数据
predictions = model(features) #计算预测结果
print(predictions.shape) # (32, 10)
tf.argmax(predictions, axis=1) #同np.argmax(), 返回预测概率最大的位置

# 计算loss
def loss(model, x, y):
    y_ = model(x)
#    y_ = tf.argmax(y_, axis=1) # 不需要吗？不需要！
    loss = tf.keras.losses.SparseCategorialCrossentropy(from_logits=False)(y, y_) #if loss_func is SparseCategorialCrossentropy
#    loss = tf.keras.losses.sparse_categorial_crossentropy(y, y_, from_logits=False) #if loss_func is sparse_categorial_crossentropy
    return loss

# 评估指标
train_loss = tf.keras.metrics.Mean('train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

test_loss = tf.keras.metrics.Mean('test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

# 每个batch的训练
def train_step(model, images, labels):
    with tf.GradientTape() as t:
        predictions = model(images)
        loss_step = loss_func(labels, predictions)
#        loss_step = loss(model, images, labels) #计算loss
    grads = t.gradient(loss_step, model.trainable_variables) #计算loss相对模型变量的梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables)) #使用grads更新模型变量，即优化过程
    train_loss(loss_step) #计算平均loss，备注：在循环过程中会记录下每个Batch的loss
    train_accuracy(labels, predictions) #计算平均accuracy

# 每个batch的预测（不用计算grads和optimizer）
def test_step(model, images, labels):
    with tf.GradientTape() as t:
        pred = model(images)
        loss_step = loss_func(labels, pred)
    test_loss(loss_step)
    test_accuracy(labels, predictions)

# 训练    
def train():
    for epoch in range(10):
        # 训练
        for (batch, (images, labels)) in enumerate(dataset):
            train_step(model, images, labels) #every batch
        print('Epoch{} is finished. loss is {}, accuracy is {}.' \
              .format(epoch, train_loss.result(), train_accuracy.result()))
        # 预测
        for (batch, (images, labels)) in enumerate(test_dataset):
            test_step(model, images, labels) #every batch
        print('Epoch{} is finished. test_loss is {}, test_accuracy is {}.' \
              .format(epoch, test_loss.result(), test_accuracy.result()))
        
        # 重制状态
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


train() #训练模型


#%% 评价指标汇总：tf.keras.metrics()
m = tf.keras.metrics.Mean('acc') #返回计算acc的对象
print(m(10))
print(m(20))
print(m([30,40]))
print(m.result().numpy()) #会保留之前的状态一起计算，返回均值25
m.reset_states() #重制状态

a = tf.keras.metrics.SparseCategoricalAccuracy('acc')
a(labels, predictions) # 自动选择概率最大位置，并计算正确率


#%% tensorboard可视化（keras定义模型）
import os
import datetime

(train_image, train_labels), (test_image, test_labels) = tf.keras.datasets.mnist.load_data()
train_image = tf.expand_dims(train_image, -1) #-1表示扩增的最后一个维度，由于使用CNN因此需要扩增数据维度
train_image = tf.cast(train_image/255, tf.float32) #需要float类型才能做梯度运算
train_labels = tf.cast(train_labels, tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels))
dataset = dataset.shuffle(10000).repeat().batch(32) # 默认repeat(1)；如果使用fit方法的话，需添加repeat(),无限循环

test_image = tf.expand_dims(test_image, -1)
test_image = tf.cast(test_image/255, tf.float32)
test_labels = tf.cast(test_labels, tf.int64)
test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_labels))
test_dataset = test_dataset.batch(32)

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16,[3,3], activation='relu',input_shape=(28,28,1)), #任意图片大小：input_shape=(None,None,1)
        tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
        tf.keras.layser.GlobalMaxPooling2D(), #GlobalAveragePooling2D()
        tf.keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

# tensorboard显示上述模型中定义的评估指标
log_dir = os.path.join('logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# tensorboard显示其他自定义指标
# 以学习速率为例，将LearningRateScheduler()传给model.fit()
file_writer = tf.summary.create_file_writer(log_dir + '/lr') #创建文件编写器
file_writer.set_as_default() #将file_writer设置成默认文件编写器

def lr_sche(epoch):
    learning_rate = 0.2
    if epoch > 5:
        learning_rate = 0.02
    elif epoch > 10:
        learning_rate = 0.01
    else:
        learning_rate = 0.005
    tf.summary.scaler('leaning_rate', data=learning_rate, step=epoch) #收集learning_rate到默认的文件编写器（即file_writer）
    return learning_rate

lr_callback = tf.keras.calllbacks.LearningRateScheduler(lr_sche) #创建lr的回调函数

model.fit(dataset, 
          epochs=10, 
          step_per_epoch=60000//128, 
          validation_data=test_data,
          validation_step=10000/128,
          callbacks=[tensorboard_callback, lr_callback])


#%% 启动tensorboard
# Jupter中启动tensorboard
%load_ext tensorboard
%matplotlib inline
%tensorboard --logdir logs

# 浏览器中启动tensorboard
# 从终端输入：tensorboard --logdir logs


#%% eager自定义训练中的tensorboard
(train_image, train_labels), (test_image, test_labels) = tf.keras.datasets.mnist.load_data()
train_image = tf.expand_dims(train_image, -1) #-1表示扩增的最后一个维度，由于使用CNN因此需要扩增数据维度
train_image = tf.cast(train_image/255, tf.float32) #需要float类型才能做梯度运算
train_labels = tf.cast(train_labels, tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels))
dataset = dataset.shuffle(10000).repeat().batch(32) # 默认repeat(1)；如果使用fit方法的话，需添加repeat(),无限循环

test_image = tf.expand_dims(test_image, -1)
test_image = tf.cast(test_image/255, tf.float32)
test_labels = tf.cast(test_labels, tf.int64)
test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_labels))
test_dataset = test_dataset.batch(32)

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16,[3,3], activation='relu',input_shape=(28,28,1)), #任意图片大小：input_shape=(None,None,1)
        tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
        tf.keras.layser.GlobalMaxPooling2D(), #GlobalAveragePooling2D()
        tf.keras.layers.Dense(10, activation='softmax')])

optimizer = tf.keras.optimizer.Adam(lr=0.01) #初始化优化器
loss_func = tf.keras.losses.SparseCategorialCrossentropy(from_logits=False) #返回一个方法，loss_func(y, y_)

train_loss = tf.keras.metrics.Mean('train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

test_loss = tf.keras.metrics.Mean('test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

def train_step(model, images, labels):
    with tf.GradientTape() as t:
        predictions = model(images)
        loss_step = loss_func(labels, predictions)
    grads = t.gradient(loss_step, model.trainable_variables) #计算loss相对模型变量的梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables)) #使用grads更新模型变量，即优化过程
    train_loss(loss_step) #计算平均loss，备注：在循环过程中会记录下每个Batch的loss
    train_accuracy(labels, predictions) #计算平均accuracy

def test_step(model, images, labels):
    with tf.GradientTape() as t:
        pred = model(images)
        loss_step = loss_func(labels, pred)
    test_loss(loss_step)
    test_accuracy(labels, predictions)

current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

train_writer = tf.summary.create_file_writer(train_log_dir)
test_writer = tf.summary.create_file_writer(test_log_dir)

  
def train():
    for epoch in range(10):
        print('Epoch is {}'. format(epoch))
        # 训练
        for (batch, (images, labels)) in enumerate(dataset):
            train_step(model, images, labels) #every batch
        with train_writer.set_as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('acc', train_accuracy.result(), step=epoch)
        print('train_end')
        
        # 预测
        for (batch, (images, labels)) in enumerate(test_dataset):
            test_step(model, images, labels) #every batch
        with test_writer.set_as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('acc', test_accuracy.result(), step=epoch)
        print('test_end')
        
        # 重制状态
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


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
# 增加网络的深度（增加卷积层、全连接层），且避免过拟合（增加训练样本、添加batch normalization、添加dropout）
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


#%% 预训练模型的使用（迁移学习）- 猫狗数据集
# keras内置预训练网络，比如VGG16、VGG19、ResNet50、Inception v3、Xception等，参考 https://keras.io/zh/applications/
# ImageNet数据集：训练集120万、验证集5万、测试集10万
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

# 使用VGG预训练网络
covn_base = keras.applications.VGG16(weight='imagenet', #weight=None 不使用预训练模型网络参数
                                     include_top=False) #include_top=False 不使用顶层全连接层的参数
                                                        #权重文件存放：/Users/tinghai/.keras/models
covn_base.summary()

# 在预训练模型基础上，添加顶层全连接层和输出层
model = keras.Sequential()
model.add(covn_base)
model.add(layers.GlobalAveragePooling2D()) #类似于Flatten()
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# 冻结预训练模型的所有参数
covn_base.trainable = False
model.summary() #可训练的参数明显减少

# 训练新添加的分类层参数
model.compile(optimizer=keras.optimizers.Adam(lr=0.0005),
              loss='binary_corssentropy',
              metrics=['acc'])
history = model.fit(train_image_ds,
                    steps_per_epoch=train_count//BATCH_SIZE,
                    epochs=12,
                    validation_data=test_image_ds,
                    validation_steps=test_count//BATCH_SIZE)

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
model.add(layers.GlobalAveragePooling2D()) #类似于Flatten()
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

# 5）联合训练
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


#%% 模型保存与恢复
# 5种：模型整体保存、模型结构保存、模型参数保存、在训练期间保存检查点（使用回调函数）、自定义训练过程中保存检查点
# mnist示例
(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
train_image = train_image/255.0 # 0-1值
test_image = test_image/255.0 # 0-1值
print(train_image.shape) # (60000, 28, 28)

ds_train_img = tf.data.Dataset.from_tensor_slices(train_image)
ds_train_lab = tf.data.Dataset.from_tensor_slices(train_label)
ds_train = tf.data.Dataset.zip((ds_train_img, ds_train_lab)) #两个tensor的对应位置元素合并，((28,28),())
ds_test = tf.data.Dataset.from_tensor_slices((test_image, test_label)) #同ds_train生成的效果一样，((28,28),())

ds_train = ds_train.shuffle(10000).repeat().batch(64)
ds_test = ds_test.batch(64) # 默认使用了repeat()

model = tf.keras.Sequential(tf.keras.layers.Flatten(input_shape=(28, 28)),
                            tf.keras.layers.Dense(128, activation='relu'),
                            tf.keras.layers.Dropout(0.1),
                            tf.keras.layers.Dense(10, activation='softmax'))
model.summary()
model.compile(opitimizer=tf.keras.optimizer.Adam(lr=0.01),
              loss='sparse_categorical_corssentropy', # 当label为顺序数字编码时
              metrics=['acc'])
history = model.fit(ds_train, 
                    epochs=5, 
                    steps_per_epoch=train_image.shape[0]//64, # "//" used to be int
                    validation_data=ds_test, 
                    validation_steps=test_image.shape[0]//64) 
model.evaluate(test_image, test_label,verbose=0) # 评估
predict = model.predict(test_image) #预测
print(predict[0]) # 返回预测概率
print(np.argmax(predict[0])) # 返回最大概率的位置
print(test_label[0]) # 真实标签

# 1）保存模型整体：包括模型结构、参数、优化器配置的保存，使得模型恢复到与保存时相同的状态
# 1.1）保存模型
model.save('./my_model.h5') #keras使用HDF5格式保存

# 1.2）加载模型
new_model = tf.keras.models.load_model('./my_model.h5') #加载模型
new_model.summary()
new_model.evaluate(test_image, test_label,verbose=0) #加载模型评估，与原模型评估结果相同

# 2）模型结构保存
json_config = model.to_json() #获取模型结构
reinitialized_model = tf.keras.model.model_from_json(json_config) #加载模型结构
reinitialized_model.summary()
reinitialized_model.evaluate(test_image, test_label,verbose=0) #报错，需要compile之后才可以
reinitialized_model.compile(opitimizer=tf.keras.optimizer.Adam(lr=0.01),
              loss='sparse_categorical_corssentropy',
              metrics=['acc'])
reinitialized_model.evaluate(test_image, test_label,verbose=0) #正确率较低，由于未经过训练

# 3）模型参数保存
weights = model.get_weights() #获取模型权重
reinitialized_model.set_weights(weights) #加载权重
reinitialized_model.evaluate(test_image, test_label,verbose=0) #正确率较高

model.save_weights('./my_weights.h5') #保存权重到磁盘
reinitialized_model.load_weights('./my_weights.h5') #从磁盘加载权重
reinitialized_model.evaluate(test_image, test_label,verbose=0) #正确率同上

# 备注：2）+3）不等同于1），由于没有保存优化器的配置，而1）保存了优化器配置！！！

# 4）在训练期间保存检查点（使用回调函数）
# 4.1）保存检查点
checkpoint_path = './my.ckpt'
my_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 moniter='val_loss',
                                                 save_best_only=False, #True,选择monitor最好的检查点
                                                 save_weights_only=True,
                                                 mode='auto',
                                                 save_freq='epoch',
                                                 verbose=0)
model = tf.keras.Sequential(tf.keras.layers.Flatten(input_shape=(28, 28)),
                            tf.keras.layers.Dense(128, activation='relu'),
                            tf.keras.layers.Dropout(0.1),
                            tf.keras.layers.Dense(10, activation='softmax'))
model.summary()
model.compile(opitimizer=tf.keras.optimizer.Adam(lr=0.01),
              loss='sparse_categorical_corssentropy', # 当label为顺序数字编码时
              metrics=['acc'])
history = model.fit(ds_train, 
                    epochs=5, 
                    steps_per_epoch=train_image.shape[0]//64,
                    validation_data=ds_test, 
                    validation_steps=test_image.shape[0]//64,
                    callbacks=[my_callback])

# 4.2）加载检查点
model = tf.keras.Sequential(tf.keras.layers.Flatten(input_shape=(28, 28)),
                            tf.keras.layers.Dense(128, activation='relu'),
                            tf.keras.layers.Dropout(0.1),
                            tf.keras.layers.Dense(10, activation='softmax'))
model.compile(opitimizer=tf.keras.optimizer.Adam(lr=0.01),
              loss='sparse_categorical_corssentropy', # 当label为顺序数字编码时
              metrics=['acc'])
model.evaluate(test_image, test_label,verbose=0) #正确率较低

# 4.2.1）加载检查点中的权重
model.load_weights(checkpoint_path) #加载检查点文件中的权重
model.evaluate(test_image, test_label,verbose=0) #加载后，正确率较高

# 4.2.2）加载检查点中的整个模型（前提回调函数中的save_weights_only=False）
model = tf.keras.models.load_model(checkpoint_path) 
model.evaluate(test_image, test_label,verbose=0) #加载后，正确率较高

# 5）自定义训练过程中保存检查点
model = tf.keras.Sequential(tf.keras.layers.Flatten(input_shape=(28, 28)),
                            tf.keras.layers.Dense(128, activation='relu'),
                            tf.keras.layers.Dropout(0.1),
                            tf.keras.layers.Dense(10, activation='softmax'))
optimizer = tf.keras.optimizers.Adam()
loss_func = tf.keras.losses.SparseCategorialCrossentropy(from_logits=True)
train_loss = tf.keras.metrics.Mean('train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

def loss(model, x, y):
    y_ = model(x)
    loss = loss_func(y, y_)
    return loss

def train_step(model, images, labels):
    with tf.GradientTape() as t:
        predictions = model(images)
        loss_step = loss_func(labels, predictions)
    grads = t.gradient(loss_step, model.trainable_variables) #计算loss相对模型变量的梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables)) #使用grads更新模型变量，即优化过程
    train_loss(loss_step) #计算平均loss，备注：在循环过程中会记录下每个Batch的loss
    train_accuracy(labels, predictions) #计算平均accuracy

# 5.1）保存检查点
cp_dir = './ckpt_dir/'
cp_prefix = os.path.join(cp_dir, 'ckpt') #文件前缀设置为ckpt
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model) #初始化检查点文件
    
def train():
    for epoch in range(10):
        for (batch, (images, labels)) in enumerate(ds_train):
            train_step(model, images, labels) #every batch
        print('Epoch{} is finished. loss is {}, accuracy is {}.' \
              .format(epoch, train_loss.result(), train_accuracy.result()))
        train_loss.reset_states()
        train_accuracy.reset_states()
        checkpoint.save(file_prefix=cp_prefix) #每个epoch保存一次检查点

train()

# 5.2）恢复检查点
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model) #初始化检查点文件
checkpoint.restore(tf.train.lastest_checkpoint(cp_dir)) #恢复最新的检查点文件
test_pred = tf.argmax(model(test_image, training=False), axis=-1).numpy()
print((test_pred == test_label).sum()/len(test_label)) #return acc


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

# 1）读取图像文件和位置文件
images = glob.glob('./images/*.jpg') 
xlms = glob.glob('./annotations/xlms/*.xlm')
print(len(images)) #7653
print(len(xlms)) #3686

names = [x.split('/')[-1].split('.')[0] for x in xlms]
imgs_train = [x for x in images if x.split('/')[-1].split('.')[0] in names] #标记位置的图片
imgs_test = [x for x in images if x.split('/')[-1].split('.')[0] not in names] #未标记位置的图片
imgs_train.sort(key=lambda x: x.split('/')[-1].split('.')[0])
xlms.sort(key=lambda x: x.split('/')[-1].split('.')[0])
print(imgs_train[-5:])
print(xlms[-5:])

# 2）获取label_datasets
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

# 3）获取image_datasets
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

# 4）恢复datasets的图片及位置框
for img, label in train_datasets.take(1): #take one batch
    plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
    out1,out2,out3,out4 = label
    xmin,ymin,xmax,ymax = out1[0].numpy()*224, out2[0].numpy()*224, out3[0].numpy()*224, out4[0].numpy()*224
    # 图片上的矩形框绘制
    rect = Rectangle((xmin, ymin),(xmax-xmin),(ymax-ymin),fill=False, color='red') #起始点坐标、x轴长度、y轴长度
    ax = plt.gca() #获取当前图像 get_current_image
    ax.axes.add_patch(rect) #当前图像中添加矩形框
    plt.show()

# 5）构建图像定位的预测模型（回归问题）
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

# 6）绘制预测效果
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

# 7）模型保存与加载
model.save('./image_location_detect_v1.h5')
new_model = tf.keras.models.load_model('./image_location_detect_v1.h5')

# 8）预测结果验证
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

# 9）图像定位的评价指标
# IoU：Intersection over Union(交并比)，即预测边框和真实边框的交集和并集的比值。

# 10）优化方向
# 先大后小：先预测出关键点，后在关键点周边预测范围
# 图片划窗：是否有关键点、关键点的位置
# 针对不确定实体个数的预测问题：先检测多个对象，然后在多个对象上回归出位置
# 变回归为分类问题：即定位区域的像素点为1，其余位置为0


#%% 自动图运算（Graph execution）
# 使用@tf.function装饰器，实现自动图运算，从而将模型转换为易于部署的tensorflow图模型
# 内部机制：
        #在eager模型关闭下，函数内代码依次运行，每个tf代码都只定义了计算节点，而非真正的计算
        #使用AutoGraph将函数中的python控制流转换成Tensorflow计算图中对应节点，比如while,for转换为tf.while，if转换为tf.cond等
        #建立函数内代码计算图，为了保证计算图的顺序，图中还会自动添加一些tf.control_dependencies节点
        #运行一次该计算图
        #基于函数类型和输入函数参数类型生成一个哈希值，并将建立的计算图缓存到一个哈希表中
        #在被@tf.function修饰的函数被再次调用时，根据函数名和输入的函数参数类型计算哈希值，检查哈希表中是否有对应计算图的缓存，如果是则继续使用已缓存的计算图，否则的话根据上述步骤建立计算图。
# 使用方法：
        #当定义多个函数实现不同运算式时，仅需要在最后调用的函数上添加@tf.function即可，这样所有的运算节点都会被编译。

@tf.function
def train_step(model, images, labels):
    pass


#%% GPU的使用和分配
import tensorflow as tf
tf.test.is_gpu_available()

# 获得当前主机上运算设备列表
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU') #[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]

# 使用既定的运算资源
tf.config.experimental.set_visible_devices(devices=gpus[0:2],device_type='GPU')

# 仅在需要时申请显存空间 (动态申请显存空间)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device_gpu, True)

# 设置消耗固定大小的显存
tf.config.experimental.set_virtual_device_configration(gpus[0],
                                                       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])


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

# 1）原图像与语义分割图像处理
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

# 2）预训练模型构建
conv_base = tf.keras.applications.VGG16(include_top=False,
                                        weigths='imagenet',
                                        input_shape=(256,256,3))
conv_base.summary()
# 最后一层(7，7，512)=>上采样为(14，14，512)=>与上层输出相加(14，14，512)=>再上采样为(28，28，256)=>与上层输出相加(28，28，256)=> ...=> 最终输出(224,224,1)

# 3）获得模型中间层的输出
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

# 4）FCN模型构建
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

# 5）FCN模型训练
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

# 6) 模型预测
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
        

#%% RNN
# tf.keras.layers.LSTM
# tf.keras.layers.GRU
# input_shape is (batch_size, seq_length, embedding_size)
# output_shape is (batch_size, tags)

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
%matplotlib inline
import numpy as np
import re

# 数据集：航空公司评论数据集
data = pd.read_csv('./Tweet.csv') #including text and label
data = data[['airline_sentiment','text']]
print(data.airline_sentiment.unique()) #标签种类
print(data.airline_sentiment.value_count()) #样本分布
data_p = data[data.airline_sentiment=='positive']
data_n = data[data.airline_sentiment=='negative']
data_n = data_n.iloc[:data_p.shape[0]]
data = pd.concat([data_p,data_n])
data = data.sample(len(data))
data['label'] = (data.airline_sentiment == 'positive').astype(int)
del data['airline_sentiment']

# 文本清洗
def reg_text(text):
    token = re.compile('[A-Za-z]+|[!?,.()]')
    new_text = token.findall(text)
    new_text = [word.lower() for word in new_text]
    return new_text

data['text'] = data['text'].apply(reg_text)

# 文本转ID
word_set = list(set([word for word in text for text in data['text'].tolist()]))
word_index = {}
for index, word in enumerate(word_set):
    word_index[word] = index + 1 #由于使用0进行填充，因此index从1开始
data_ok = data['text'].apply(lambda x: [word_index.get(word,0) for word in x])

# 文本长度分析
text_len = data_ok.apply(lambda x : len(x))
print(text_len.describe())
max_len = max(text_len) #40
max_word = len(word_set) + 1 #1为填充
data_ok = tf.keras.preprocessing.sequence.pad_sequences(data_ok.values, max_len) #填充0,使得长度为max_len

# LSTM模型构建
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(max_word, 50, input_length=max_len)) #向量化，input_length输入数据的长度, (None, max_len, 50)
model.add(tf.keras.layers.LSTM(64))
#model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))) #添加L2正则化
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
              loss='binary_crossentropy',
              metrics=['acc'])
model.fit(data_ok, data.labels.values, epochs=10, batch_size=32, validation_split=0.2) #选择20%作为测试集  


#%% RNN - 北京空气污染预测
# 目标：预测未来的PM2.5水平

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
%matplotlib inline
import numpy as np
import pandas as pd
import re
import datetime

# 1）数据处理
data = pd.read_csv('./dataset/PRSA_data.csv') #label is PM2.5

# Nan数据处理
data['pm2.5'].isna().sum() #2067, 由于去掉PM2.5为空的序列会打乱数据的时序性，因此采取填充的方式
data = data.iloc[24:] #去掉前24个pm2.5为空的数据
data = data.fillna(method='ffill') #前向填充，使用前天的数据填充昨天的数据

# 时间列合并成一个索引值
data['tm'] = data.apply(lambda x: datetime.datetime(year=x['year'], 
                                       month=x['month'], 
                                       day=x['day'], 
                                       hour=x['hour']), axis=1)
data.drop(columns=['year','month','day','hour','No'], inplace=True)
data = data.set_index('tm')

# 非数字列处理
print(data.cbwd.unique())
data = data.join(pd.get_dummies(data.cbwd)) #cbwd列进行onehot编码后，与原dataframe进行拼接
del data.cbwd

# 数据时序性采样
# 使用前面多久的时序数据，来预测接下来多久的数据？
data['pm2.5'][-1000:].plot() # 最后1000次PM2.5观测情况

seq_length = 5 * 24 #使用当前点前5天的数据
delay = 24 #预测当前点后1天的数据，因此delay=24h

data_ = []
for i in range(len(data) - seq_length - delay):
    data_.append(data.iloc[i: i + seq_length + delay]) #按时序依次采样6天的数据

data_ = np.array([df.values for df in data_]) #转化成numpy形式
print(data_.shape) #(43656,144,11)，一共采样出43656条时序数据，每条时序包括144个时间点，每个时间点包含11个特征值

# 训练和测试数据生成
np.random.shuffle(data_)
x = data_[:,:seq_length,:] #(43656,120,11)
y = data_[:,-1,0] #(43656,)，-1 => 取每条时序的最后一个时间点，0 => 最后一个时间点的pm2.5值
split_b = int(0.8 * data_.shape[0])
train_x = x[:split_b]
train_y = y[:split_b]
test_x = x[split_b:]
test_y = y[split_b:]

# 数据标准化 
# 注意：a. 训练集需要单独进行标准化，不能使用全局数据的标准化结果
# b. 测试集也应该使用训练集的均值和标准差进行标准化
# c. 预测值是否需要进行标准化呢？不需要
mean = train_x.mean(axis=0) #按列计算均值
std = train_x.std(axis=0) #按列计算标准差
train_x = (train_x - mean)/std
test_x = (test_x - mean)/std

# 2）构建全连接神经网络
BATCH_SIZE = 32
model = tf.keras.Sequentail()
model.add(tf.keras.Flatten(input_shape=train_x.shape[1:]))
model.add(tf.keras.Dense(32,activation='relu'))
model.add(tf.keras.Dense(1)) #回归问题，无需激活函数
model.compile(optimizer='adam', 
              loss='mse', 
              metrics=['mae'])
history = model.fit(train_x,train_y, 
                    epochs=50, 
                    steps_per_epoch=split_b//BATCH_SIZE,
                    validation_data=(test_x,test_y), 
                    validation_steps=(data_.shape[0]-split_b)//BATCH_SIZE)
plt.plot(history.epoch, history.history['mean_absolute_error'], 'r', label='train-loss')
plt.plot(history.epoch, history.history['val_mean_absolute_error'], 'g', label='validation-loss')
plt.legend()

# 3）构建单层LSTM网络
model = tf.keras.Sequentail()
model.add(tf.keras.layers.LSTM(units=32,
                               input_shape=train_x.shape[1:],
                               activation='tanh')) #默认return_sequences=False, 即只返回LSTM的output结果，而非state结果
model.add(tf.keras.Dense(1))
model.compile(optimizer='adam', 
              loss='mse', 
              metrics=['mae'])
history = model.fit(train_x,train_y, 
                    epochs=150, 
                    steps_per_epoch=split_b//BATCH_SIZE,
                    validation_data=(test_x,test_y), 
                    validation_steps=(data_.shape[0]-split_b)//BATCH_SIZE)
plt.plot(history.epoch, history.history['mean_absolute_error'], 'r', label='train-loss')
plt.plot(history.epoch, history.history['val_mean_absolute_error'], 'g', label='validation-loss')
plt.legend()

# 4）构建多层LSTM网络
model = tf.keras.Sequentail()
model.add(tf.keras.layers.LSTM(units=32,
                               input_shape=train_x.shape[1:],
                               activation='tanh',
                               return_sequences=True)) #返回LSTM的output和state结果
model.add(tf.keras.layers.LSTM(units=32,
                               activation='tanh',
                               return_sequences=True))
model.add(tf.keras.layers.LSTM(units=32,
                               activation='tanh',
                               return_sequences=False)) #最后一个LSTM，只使用output输出，用于连接Dense层
model.add(tf.keras.Dense(1))

# 添加回调函数，在训练过程中降低学习速率
#在连续3个epoch中val_loss没有降低，则降低LR为原来的0.1倍，但最小不超过0.00001
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 patience=3, 
                                                 factor=0.1, 
                                                 min_lr=0.00001)
model.compile(optimizer='adam', 
              loss='mse', 
              metrics=['mae'])
history = model.fit(train_x,train_y, 
                    epochs=200, 
                    steps_per_epoch=split_b//BATCH_SIZE,
                    validation_data=(test_x,test_y), 
                    validation_steps=(data_.shape[0]-split_b)//BATCH_SIZE,
                    callbacks=[lr_reduce])
plt.plot(history.epoch, history.history['mean_absolute_error'], 'r', label='train-loss')
plt.plot(history.epoch, history.history['val_mean_absolute_error'], 'g', label='validation-loss')
plt.legend()







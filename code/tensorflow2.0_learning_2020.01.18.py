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
# pip install tensorflow==2.0.0-beta1


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
#    img_tensor = img_tensor.resize_image_with_crop_or_pad(img_tensor) #图像resize后不变形
    img_tensor = img_tensor.resize(img_tensor, (256,256)) #图像可能发生变形，使用resize可以使得解析后的tensor具备shape
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


#%% 文本向量化
from tensorflow import keras
from tensorflow.keras import layers

# 电影评论数据
data = keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = data.load_data(num_words=10000)
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
model.add(layers.Dense(128, activation='relu'), kernel_regularizer=regularizers.l2(0.01)) #添加L2正则化
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


#%% mnist-cnn-eager模式
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

test_loss = tf.keras.metrics.Mean('train_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

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
print(m.result().numpy()) #会保留之前的状态一起计算，返回 25
m.reset_states() #重制状态

a = tf.keras.metrics.SparseCategoricalAccuracy('acc')
a(labels, predictions) # 自动选择概率最大位置，并计算正确率


#%% tensorboard可视化
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

log_dir = os.path.join('logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(dataset, 
          epochs=10, 
          step_per_epoch=60000//128, 
          validation_data=test_data,
          validation_step=10000/128,
          callbacks=[tensorboard_callback])

# Jupter中启动tensorboard
%load_ext tensorboard
%matplotlib inline
%tensorboard --logdir logs

# 浏览器中启动tensorboard
# 从终端启动：tensorboard --logdir logs









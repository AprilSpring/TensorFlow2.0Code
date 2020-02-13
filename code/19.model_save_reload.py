#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 18:30:11 2020

模型的保存与恢复:
    模型整体：h5文件
    模型参数：h5文件
    模型结构：json文件
    检查点文件：ckpt文件，包括model.fit()中使用回调函数 和 自定义训练模型
    savedmodel：pb文件

@author: tinghai
"""

import tensorflow as tf
from tensorflow import keras    
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np 
import glob
import os
import pathlib
import random

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

#%% 1）保存模型整体：包括模型结构、参数、优化器配置的保存，使得模型恢复到与保存时相同的状态
# 1.1）保存模型
model.save('./my_model.h5') #keras使用HDF5格式保存

# 1.2）加载模型
new_model = tf.keras.models.load_model('./my_model.h5') #加载模型
new_model.summary()
new_model.evaluate(test_image, test_label,verbose=0) #加载模型评估，与原模型评估结果相同

#%% 2）模型结构保存
json_config = model.to_json() #获取模型结构
reinitialized_model = tf.keras.model.model_from_json(json_config)
reinitialized_model.summary()
reinitialized_model.evaluate(test_image, test_label,verbose=0) #报错，需要compile之后才可以
reinitialized_model.compile(opitimizer=tf.keras.optimizer.Adam(lr=0.01),
              loss='sparse_categorical_corssentropy',
              metrics=['acc'])
reinitialized_model.evaluate(test_image, test_label,verbose=0) #正确率较低，由于未经过训练

#%% 3）模型参数保存
weights = model.get_weights() #获取模型权重
reinitialized_model.set_weights(weights) #加载权重
reinitialized_model.evaluate(test_image, test_label,verbose=0) #正确率较高

model.save_weights('./my_weights.h5') #保存权重到磁盘
reinitialized_model.load_weights('./my_weights.h5') #从磁盘加载权重
reinitialized_model.evaluate(test_image, test_label,verbose=0) #正确率同上

# 备注：2）+3）不等同于1），由于没有保存优化器的配置，而1）保存了优化器配置！！！

#%% 4）在训练期间保存检查点（使用回调函数）
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


#%% 5）自定义训练过程中保存检查点
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
# cp_prefix = os.path.join(cp_dir, 'ckpt') #文件前缀设置为ckpt
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model) #初始化检查点文件
manager = tf.train.CheckpointManager(checkpoint, directory=cp_dir, checkpoint_name='model.ckpt', max_to_keep=3)

def train():
    for epoch in range(10):
        for (batch, (images, labels)) in enumerate(ds_train):
            train_step(model, images, labels)
        print('Epoch{} is finished. loss is {}, accuracy is {}.' \
              .format(epoch, train_loss.result(), train_accuracy.result()))
        train_loss.reset_states()
        train_accuracy.reset_states()
        # checkpoint.save(file_prefix=cp_prefix) #每个epoch保存一次检查点
        manager.save(checkpoint_number=epoch)

train()

# 5.2）恢复检查点
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model) #初始化检查点文件
checkpoint.restore(tf.train.lastest_checkpoint(cp_dir)) #恢复最新的检查点文件
test_pred = tf.argmax(model(test_image, training=False), axis=-1).numpy()
print((test_pred == test_label).sum()/len(test_label)) #return acc


# 6）savedmodel保存和加载
# 与前面介绍的Checkpoint不同，SavedModel包含了一个TensorFlow程序的完整信息,不仅包含参数的权值，
# 还包含计算的流程（即计算图），参考：https://blog.csdn.net/zkbaba/article/details/104238916

# 6.1）使用tf.keras.models.Sequential构建模型
# tf.saved_model.save(model,'保存的目标文件夹名称') 和 tf.saved_model.load('保存的目标文件夹名称')
import tensorflow as tf

num_epochs = 1
batch_size = 50
learning_rate = 0.001

# 模型构建
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
    ])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
model.fit(ds_train, epochs=num_epochs, batch_size=batch_size)
tf.saved_model.save(model, "saved/1")

# 模型调用和测试
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
y_pred = model(test_image) # or model.predict()
sparse_categorical_accuracy.update_state(y_true=test_label, y_pred=y_pred)
print("test accuracy: %f" % sparse_categorical_accuracy.result())

# 6.2）继承tf.keras.Model构建模型
# 要点：a. call()方法需要以 @tf.function 修饰，以转化为 SavedModel 支持的计算图
#      b. 模型推断时需要显式调用model.call()方法

# 继承tf.keras.Model的模型构建
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    @tf.function                    # call()方法需要以 @tf.function 修饰，以转化为 SavedModel 支持的计算图
    def call(self, inputs):         # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)    # [batch_size, 784]
        x = self.dense1(x)          # [batch_size, 100]
        x = self.dense2(x)          # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output

# 模型调用
model = MLP()

# 模型保存
tf.saved_model.save(model, "saved/1")

# 模型调用
new_model = tf.saved_model.load("saved/1")

# 模型测试
y_pred = new_model.call(test_image) #针对继承tf.keras.Model的模型，需要使用model.call()


















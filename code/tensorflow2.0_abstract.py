#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:42:44 2020

tensorflow2.0学习重点

@author: tinghai
"""

#%% install
pip install tensorflow==2.0.0-beta1
pip install tensorflow-gpu==2.0.0-beta0

import tensorflow as tf

tf.test.is_gpu_avialble()


#%% tf.keras
tf.keras.Sequential()

tf.keras.Input()

tf.keras.utils.to_categorial() #one-hot

tf.keras.Model(inputs=inputs, outputs=outputs)

tf.keras.models.Model()
tf.keras.models.model_from_json()

tf.keras.layers.Flatten()
tf.keras.layers.Dense()
tf.keras.layers.concatenate()
tf.keras.layers.Conv2D()
tf.keras.layers.Batchnormalization()
tf.keras.layers.MaxPooling2D()
tf.keras.layers.UpSampling2D()
tf.keras.layers.Dropout()
tf.keras.layers.GlobalAveragePooling2D()
tf.keras.layers.Conv2DTranspose()
tf.keras.layers.LSTM()
tf.keras.layers.GRU()
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_num, return_sequences=True)) #bilstm

tf.keras.preprocessing.image.array_to_img()
tf.keras.preprocessing.text.Tokenizer()
tf.keras.preprocessing.sequence.pad_sequences()

tf.keras.optimizers.Adam()

tf.keras.losses.SparseCategorialCrossentropy() #返回一个方法，loss_func(y, y_)
tf.keras.losses.sparse_categorial_crossentropy(y_true, y_pred, from_logits = False)
tf.keras.losses.BinaryCrossentropy()

tf.keras.metrics.Mean('train_loss')
tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
tf.keras.calllbacks.LearningRateScheduler()
tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                   moniter='val_loss',
                                   save_best_only=False, #True,选择monitor最好的检查点进行保存
                                   save_weights_only=True,
                                   mode='auto',
                                   save_freq='epoch',
                                   verbose=0)

tf.keras.applications.xception.Xception(include_top=False,
                                        weigths='imagenet',
                                        input_shape=(224,224,3),
                                        pooling='avg')
tf.keras.applications.VGG16(include_top=False,
                            weigths='imagenet',
                            input_shape=(256,256,3))
tf.keras.applications.MobileNetV2(include_top=False,
                                  weigths=None, #仅使用MobileNetV2的架构，没有使用权重
                                  input_shape=(224,224,3))

tf.keras.estimator.model_to_estimator(model, config=config)


#%% model
model.add()
model.summary()
model.compile()
model.fit()
model.evaluate()
model.predict()
y_ = model(x) #函数式API调用
model.trainable_variables #模型可训练参数
model = MLP() #MLP为继承tf.keras.Model的模型
y_pred = model.call(test_image)


#%% tf.models
base_model = VGG16(weights='imagenet', include_top=True)
model = tf.models.Model(inputs=base_model.input,
                        outputs=base_model.get_layer('block4_pool').output) #提取子模型


#%% model save and reload
model.save('./my_model.h5')  #模型整体保存
new_model = tf.keras.models.load_model('./my_model.h5') #加载模型

json_config = model.to_json() #获取模型结构
model = tf.keras.models.model_from_json(json_config) #加载模型结构

with open('./model.json', 'w') as json_file: #模型结构写出
    json_file.write(json_config)
model = tf.keras.models.model_from_json(open('./model.json').read()) #模型结构读入

weights = model.get_weights() #获取模型权重
model.set_weights(weights) #加载模型权重

model.save_weights('./my_weights.h5') #模型权重写出
model.load_weights('./my_weights.h5') #模型权重读入

model = tf.keras.models.load_model(checkpoint_path) #加载检查点文件
model.load_weights(checkpoint_path) #加载检查点文件中的权重


#%% checkpoint
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model) #初始化检查点文件
checkpoint.save(file_prefix=cp_prefix) #每个epoch保存一次检查点
checkpoint.restore(tf.train.lastest_checkpoint(cp_dir)) #恢复最新的检查点文件


#%% savedmodel
tf.saved_model.save(model, "saved/1")
tf.saved_model.load( "saved/1")


#%% 预训练模型
conv_base.get_layer('block5_conv3').output
conv_base.trainable = False


#%% history
history.epoch
history.history.get('acc')


#%% tf.GradientTape
with tf.GradientTape() as t:
    predictions = model(images)
    loss_step = tf.keras.losses.SparseCategorialCrossentropy(from_logits=False)(labels, predictions)

grads = t.gradient(loss_step, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))

train_loss = tf.keras.metrics.Mean('train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

train_loss(loss_step)
train_accuracy(labels, predictions) 

train_loss.result()
train_accuracy.result()

train_loss.reset_states()
train_accuracy.reset_states()


#%% tf.data
tf.data.Dataset.from_tensor_slices()
tf.data.Dataset.zip()
AUTOTUNE = tf.data.experimental.AUTOTUNE


#%% datasets
dataset.shuffle()
dataset.repeat()
dataset.batch()
dataset.map(func, num_parallel_calls=AUTOTUNE)
dataset.filter(func)
dataset.skip()
dataset.take()
dataset.prefetch(AUTOTUNE)
image_batch,label_batch = next(iter(dataset))
dataset.cache()
dataset.padded_batch(BATCH_SIZE, padded_shapes=([40], [40])) #填充为最大长度40


#%% tf.image
tf.image.decode_jpeg()
tf.image.decode_png()
tf.image.resize()
tf.image.random_crop()
tf.image.random_flip_left_right(image)
tf.image.random_flip_up_down(image)
tf.image.random_brigtness(image, 0.5)
tf.image.random_contrast(image, 0, 1)
tf.image.random_hue(image, max_delta=0.3)
tf.image.random_saturation(image, lower=0.2, upper=1.0)


#%% tf.summary
train_writer = tf.summary.create_file_writer(train_log_dir)

with train_writer.set_as_default():
    tf.summary.scalar('loss', train_loss.result(), step=epoch)


#%% tf.config
tf.config.experimental.list_physical_devices()
tf.config.experimental.set_visible_devices()
tf.config.experimental.set_memory_growth()
tf.config.experimental.set_virtual_device_configration()


#%% tf.estimator
model = tf.estimator.LinearRegressor(featcols) #回归模型


#%% crf
import tensorflow_addons as tf_ad
log_likelihood, transition_params = tf_ad.text.crf_log_likelihood(logits, label_sequences, text_lens) # train step, log_likelihood用于计算loss
viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params) #predict step


#%% tf.其他
tf.io.read_file(img_path)
tf.cast()
tf.expand_dims()
tf.reshape()
tf.argmax()
tf.logical_and(True, False) #False
tf.logical_or(True, False) #True
tf.py_function(func=encode, inp=[pt, en], Tout=[tf.int64, tf.int64])
tf.nn.softmax()


#%% plt
plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
plt.plot()
plt.legend()


#%% display
import Ipython.display as display
display.display(display.Image(image_path))


#%% pathlib
data_root = pathlib.Path(data_dir)
data_root.iterdir()
list(data_root.glob('*/*'))
sorted(item.name for item in data_root.glob('*/'))


#%% glob
glob.glob('./test/*.jpg')


#%% etree
from lxml import etree
xml = open(path).read()
sel = etree.HTML(xml)
width = int(sel.xpath('//size/width/text()')[0])






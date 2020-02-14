#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:32:04 2020

tf.keras文本向量化

@author: tinghai
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline
# !pip install tensorflow==2.0.0-beta1
# !pip install tensorflow-gpu==2.0.0-beta0

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
model.add(layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))#添加L2正则化
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
              loss='binary_crossentropy',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=256, validation_data=(x_test, y_test))


#%% 文本转ID
# 在进行embedding前，需要对文本进行ID转化
# download and read data into data structures
labeled_sentences = download_and_read(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip")
sentences = [s for (s, l) in labeled_sentences]
labels = [int(l) for (s, l) in labeled_sentences]

# tokenize sentences
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_counts)
print("vocabulary size: {:d}".format(vocab_size))

word2idx = tokenizer.word_index
idx2word = {v:k for (k, v) in word2idx.items()}

# create dataset
max_seqlen = 64
sentences_as_ints = tokenizer.texts_to_sequences(sentences)
sentences_as_ints = tf.keras.preprocessing.sequence.pad_sequences(
    sentences_as_ints, maxlen=max_seqlen)
labels_as_ints = np.array(labels)
dataset = tf.data.Dataset.from_tensor_slices((sentences_as_ints, labels_as_ints))

# model construct
# ...

# predict on batches
from sklearn.metrics import accuracy_score, confusion_matrix

labels, predictions = [], []
idx2word[0] = "PAD"
is_first_batch = True
for test_batch in test_dataset:
    inputs_b, labels_b = test_batch
    pred_batch = best_model.predict(inputs_b)
    predictions.extend([(1 if p > 0.5 else 0) for p in pred_batch])
    labels.extend([l for l in labels_b])
    if is_first_batch:
        for rid in range(inputs_b.shape[0]):
            words = [idx2word[idx] for idx in inputs_b[rid].numpy()]
            words = [w for w in words if w != "PAD"]
            sentence = " ".join(words)
            print("{:d}\t{:d}\t{:s}".format(labels[rid], predictions[rid], sentence))
        is_first_batch = False

print("accuracy score: {:.3f}".format(accuracy_score(labels, predictions)))
print("confusion matrix")
print(confusion_matrix(labels, predictions))



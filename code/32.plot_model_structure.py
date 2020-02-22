#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Created on: 2020/2/13 6:45 下午

@annotaion:
    绘制模型的结构：tf.keras.utils.plot_model

@author: tinghai
"""

import tensorflow as tf

# 构建一个共享embedding层的网络
def build_model():
    # Variable-length sequence of integers
    text_input_a = tf.keras.Input(shape=(None,), dtype='int32')

    # Variable-length sequence of integers
    text_input_b = tf.keras.Input(shape=(None,), dtype='int32')

    # Embedding for 1000 unique words mapped to 128-dimensional vectors
    shared_embedding = tf.keras.layers.Embedding(1000, 128)

    # We reuse the same layer to encode both inputs
    encoded_input_a = shared_embedding(text_input_a)
    encoded_input_b = shared_embedding(text_input_b)

    #two logistic predictions at the end
    prediction_a = tf.keras.layers.Dense(1, activation='sigmoid', name='prediction_a')(encoded_input_a)
    prediction_b = tf.keras.layers.Dense(1, activation='sigmoid', name='prediction_b')(encoded_input_b)

    # this model has 2 inputs, and 2 outputs
    # in the middle we have a shared model
    model = tf.keras.Model(inputs=[text_input_a, text_input_b],
                           outputs=[prediction_a, prediction_b])

    tf.keras.utils.plot_model(model, to_file="./shared_model.png")

build_model()


#%%
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model

plot_model(model, show_shapes=True, to_file='model_simple.png')
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Created on: 2020/2/14 1:59 下午

@annotaion:
    回归模型：预测波士顿房价

@author: tinghai
"""
import tensorflow as tf

featcols = [
    tf.feature_column.numeric_column("area"),
    tf.feature_column.categorical_column_with_vocabulary_list("type",["bungalow","apartment"])
]


def train_input_fn():
    features = {"area":[1000,2000,4000,1000,2000,4000],
                "type":["bungalow","bungalow","house",
                        "apartment","apartment","apartment"]}
    labels = [ 500 , 1000 , 1500 , 700 , 1300 , 1900 ]
    return features, labels

model = tf.estimator.LinearRegressor(featcols)
model.train(train_input_fn, steps=200)


# 预测
def predict_input_fn():
    features = {"area":[1500,1800], "type":["house","apartment"]}
    return features

predictions = model.predict(predict_input_fn)

print(next(predictions))
print(next(predictions))


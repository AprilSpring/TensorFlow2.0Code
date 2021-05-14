# -*- coding: utf-8 -*-
# @Time    : 2020-10-28 10:11
# @Author  : WenYi
# @Contact : 1244058349@qq.com
# @Description :  ESMM model for CTR and CVR predict task

import tensorflow as tf
# 是否使用GPU
tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)
from tensorflow.keras.models import Model
from tensorflow.keras import layers

class CTCVRNet:
	def __init__(self, user_feature_dim, item_feature_dim):
		self.user_feature_dim = user_feature_dim
		self.item_feature_dim = item_feature_dim

	def base_model(self, user_feature, item_feature):
		user_feature = layers.Dense(128, activation='relu')(user_feature)
		user_feature = layers.Dense(64, activation='relu')(user_feature)
		user_feature = layers.Dropout(0.5)(user_feature)
		user_feature = layers.BatchNormalization()(user_feature)

		item_feature = layers.Dense(128, activation='relu')(item_feature)
		item_feature = layers.Dense(64, activation='relu')(item_feature)
		item_feature = layers.Dropout(0.5)(item_feature)
		item_feature = layers.BatchNormalization()(item_feature)

		dense_feature = layers.concatenate([user_feature, item_feature], axis=-1)
		dense_feature = layers.Dropout(0.5)(dense_feature)
		dense_feature = layers.BatchNormalization()(dense_feature)
		dense_feature = layers.Dense(64, activation='relu')(dense_feature)

		pred = layers.Dense(1, activation='sigmoid')(dense_feature)
		return pred

	# def build_ctr_model(self,ctr_user_feature, ctr_item_feature):
	# 	pred = self.base_model(ctr_user_feature, ctr_item_feature)
	# 	return pred
	#
	# def build_cvr_model(self,cvr_user_feature, cvr_item_feature):
	# 	pred = self.base_model(cvr_user_feature, cvr_item_feature)
	# 	return pred
	
	def build(self):
		# CTR model input
		ctr_user_feature = layers.Input(shape=(self.user_feature_dim,))
		ctr_item_feature = layers.Input(shape=(self.item_feature_dim,))

		# CVR model input
		cvr_user_feature = layers.Input(shape=(self.user_feature_dim,))
		cvr_item_feature = layers.Input(shape=(self.item_feature_dim,))

		ctr_pred = self.base_model(ctr_user_feature, ctr_item_feature)
		cvr_pred = self.base_model(cvr_user_feature, cvr_item_feature)
		ctcvr_pred = tf.multiply(ctr_pred, cvr_pred)

		model = Model(
			inputs=[ctr_user_feature, ctr_item_feature, cvr_user_feature, cvr_item_feature],
			outputs=[ctr_pred, ctcvr_pred])
		
		return model

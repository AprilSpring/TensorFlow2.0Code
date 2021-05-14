# -*- coding: utf-8 -*-
# @Time    : 2020-11-05 17:41
# @Author  : WenYi
# @Contact : wenyi@cvte.com
# @Description :  script description


import tensorflow as tf
import os
import time

tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import *
from esmm import CTCVRNet

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"


def train_model(train_data, train_label, val_data, val_label, user_feature_dim, item_feature_dim, modelpath):
	ctcvr = CTCVRNet(user_feature_dim, item_feature_dim)
	ctcvr_model = ctcvr.build()
	opt = optimizers.Adam(lr=0.003, decay=0.0001)
	ctcvr_model.compile(optimizer=opt, loss=["binary_crossentropy", "binary_crossentropy"], loss_weights=[1.0, 1.0],
	                    metrics=[tf.keras.metrics.AUC()])

	# call back function
	checkpoint = ModelCheckpoint(
		modelpath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	reduce_lr = ReduceLROnPlateau(
		monitor='val_loss', factor=0.8, patience=2, min_lr=0.0001, verbose=1)
	earlystopping = EarlyStopping(
		monitor='val_loss', min_delta=0.0001, patience=8, verbose=1, mode='auto')
	callbacks = [checkpoint, reduce_lr, earlystopping]

	# model train
	ctcvr_model.fit(train_data, train_label,
					batch_size=256,
					epochs=50,
	                validation_data=(val_data,val_label),
					callbacks=callbacks,
	                verbose=0,
	                shuffle=True)
	
	# save as tf_serving model
	saved_model_path = './esmm/{}'.format(int(time.time()))
	# ctcvr_model = tf.keras.models.load_model('esmm_best.h5')
	tf.saved_model.save(ctcvr_model, saved_model_path)
	return ctcvr_model

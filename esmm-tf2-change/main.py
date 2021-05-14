# -*- coding: utf-8 -*-
# @Time    : 2020-11-05 17:49
# @Author  : WenYi
# @Contact : wenyi@cvte.com
# @Description :  script description

import numpy as np
import pandas as pd
from esmm import CTCVRNet
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from train import train_model
# from importlib import reload
# reload(train)

ctr_user_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['user_numerical_{}'.format(i) for i in range(5)])
ctr_item_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['item_numerical_{}'.format(i) for i in range(5)])
cvr_user_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['user_numerical_{}'.format(i) for i in range(5)])
cvr_item_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['item_numerical_{}'.format(i) for i in range(5)])

ctr_user_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                              columns=['user_numerical_{}'.format(i) for i in range(5)])
ctr_item_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                              columns=['item_numerical_{}'.format(i) for i in range(5)])
cvr_user_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                              columns=['user_numerical_{}'.format(i) for i in range(5)])
cvr_item_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                              columns=['item_numerical_{}'.format(i) for i in range(5)])

ctr_target_train = pd.DataFrame(np.random.randint(0, 2, size=10000))
cvr_target_train = pd.DataFrame(np.random.randint(0, 2, size=10000))

ctr_target_val = pd.DataFrame(np.random.randint(0, 2, size=10000))
cvr_target_val = pd.DataFrame(np.random.randint(0, 2, size=10000))

train_data = [ctr_user_feature_train,ctr_item_feature_train,cvr_user_feature_train,cvr_item_feature_train]
train_label = [ctr_target_train,cvr_target_train]

val_data = [ctr_user_feature_val,ctr_item_feature_val,cvr_user_feature_val,cvr_item_feature_val]
val_label = [ctr_target_val, cvr_target_val]

pred_data = [ctr_user_feature_train.iloc[0:20], ctr_item_feature_train.iloc[0:20],cvr_user_feature_train.iloc[0:20], cvr_item_feature_train.iloc[0:20]]

user_feature_dim = ctr_user_feature_train.shape[1]
item_feature_dim = ctr_item_feature_train.shape[1]

modelpath = "esmm_best.h5"

# trian model
ctcvr_model = train_model(train_data, train_label, val_data, val_label, user_feature_dim, item_feature_dim, modelpath)

# load model
# ctcvr_model = tf.keras.models.load_model('esmm_best.h5')

# model predict
[ctr_pred, ctcvr_pred] = ctcvr_model.predict(pred_data)

# get cvr predict
cvr_pred = ctcvr_pred/ctr_pred


# 参考：https://github.com/busesese/ESMM
# 问题：实现这个模型的时候怎么训练，损失函数怎么写，数据怎么构造？
# 这里我们可以看到主任务是CVR任务，副任务是CTR任务，实际生产的数据是用户曝光数据，点击数据和转化数据，
# 那么曝光和点击数据可以构造副任务的CTR模型（正样本：曝光&点击、负样本：曝光&未点击），
# 曝光和转化数据(转化必点击)构造的是CTCVR任务（正样本：点击&转化、负样本：点击&未转化），
# 模型的输出有3个，CTR模型输出预测的pCTR,CVR模型输出预测的pCVR,联合模型输出预测的pCTCVR=pCTR*pCVR，
# 由于CVR模型的输出标签不好直接构造，因此这里损失函数loss = ctr的损失函数 + ctcvr的损失函数，
# 因为pctcvr=pctr*pcvr，所以loss中也充分利用到CVR模型的参数。

# 综上,
# 模型的Input分为2个数据集：1）CTR任务数据、2）CTCVR任务数据
# 模型的Output也是2个预测结果：1）pCTR、2）pCTCVR
# 而pCVR = pCTCVR / pCTR




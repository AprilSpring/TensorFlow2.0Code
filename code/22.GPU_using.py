#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:51:51 2020

GPU的使用和分配

@author: tinghai
"""

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
    tf.config.experimental.set_memory_growth(device=gpu, True)

# 设置消耗固定大小的显存
tf.config.experimental.set_virtual_device_configration(gpus[0],
                                                       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])





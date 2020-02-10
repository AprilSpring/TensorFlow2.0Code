#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:37:41 2020

tensorboard启动

@author: tinghai
"""


#%% 启动tensorboard
# Jupter中启动tensorboard
%load_ext tensorboard
%matplotlib inline
%tensorboard --logdir logs

# 浏览器中启动tensorboard
# 从终端输入：tensorboard --logdir logs


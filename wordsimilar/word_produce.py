#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    @Time : 2021/4/25 7:09 下午
    @Author : liutingting
    @File : word_produce.py
    @Used : 
'''

import codecs
# from arff import unichr
import os
import pygame

## 1）文字生成
start,end = (0x4E00, 0x9FA5)  #汉字编码的范围
with codecs.open("chinese.txt", "wb", encoding="utf-8") as f:
 for codepoint in range(int(start),int(end)):
  f.write(chr(codepoint))  #写出汉字


## 2）文字写出图片
chinese_dir = 'chinese'
if not os.path.exists(chinese_dir):
    os.mkdir(chinese_dir)

pygame.init()
start,end = (0x4E00, 0x9FA5) # 汉字编码范围
for codepoint in range(int(start),int(end)):
    # word = unichr(codepoint)
    word = chr(codepoint)
    font = pygame.font.Font("MSYH.TTC", 32)
    # 当前目录下要有微软雅黑的字体文件msyh.ttc,或者去c:\Windows\Fonts目录下找
    # 64是生成汉字的字体大小
    rtext = font.render(word, True, (0, 0, 0), (255, 255, 255))
    pygame.image.save(rtext, os.path.join(chinese_dir, word + ".png"))




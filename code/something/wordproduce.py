#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Created on: 2021/4/25 10:55 下午

@annotaion: https://www.jb51.net/article/155544.htm

@author: tinghai
"""


import codecs
import os
import pygame

start,end = (0x4E00, 0x9FA5) #汉字编码的范围
with codecs.open("../data/chinese.txt", "wb", encoding="utf-8") as f:
    for codepoint in range(int(start),int(end)):
        f.write(chr(codepoint)) #写出汉字


chinese_dir = '../data/chinese'
if not os.path.exists(chinese_dir):
    os.mkdir(chinese_dir)

pygame.init()
start, end = (0x4E00, 0x9FA5)  # 汉字编码范围
for codepoint in range(int(start), int(end)):
    word = chr(codepoint)
    font = pygame.font.Font("MSYH.TTC", 32)
    # 当前目录下要有微软雅黑的字体文件msyh.ttc,或者去c:\Windows\Fonts目录下找
    # 64是生成汉字的字体大小
    rtext = font.render(word, True, (0, 0, 0), (255, 255, 255))
    pygame.image.save(rtext, os.path. join(chinese_dir, word + ".png"))






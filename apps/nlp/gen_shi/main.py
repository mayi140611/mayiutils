#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: main.py
@time: 2019-05-07 19:55
"""
#! -*- coding:utf-8 -*-
# 一个简单的基于VAE和CNN的作诗机器人
# 来自：https://kexue.fm/archives/5332

import re
import codecs
import numpy as np
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import Callback


n = 5 # 只抽取五言诗
latent_dim = 64 # 隐变量维度
hidden_dim = 64 # 隐层节点数
if __name__ == '__main__':
    s = codecs.open('/Users/luoyonggui/Documents/datasets/nlp/shi.txt', encoding='utf-8').read()

    # 通过正则表达式找出所有的五言诗
    s = re.findall(r'(.{%s}，.{%s}。)'%(n,n), s)
    print(s[:2], len(s))
    """
    ['秦川雄帝宅，函谷壮皇居。', '绮殿千寻起，离宫百雉馀。'] 149754
    """

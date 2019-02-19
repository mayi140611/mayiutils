#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: estest.py
@time: 2019/1/21 18:26
"""
from mayiutils.nlp.jieba_wrapper import JieBaWrapper as jbw
import json

if __name__ == '__main__':
    r = jbw.lcut('我爱李小福', HMM=False)
    print(r)
    jbw.loadUserDict('./mayiutils/nlp/jieba_userdict.txt')
    r = jbw.lcut('我爱李小福', HMM=False)
    print(r)





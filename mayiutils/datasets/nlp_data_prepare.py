#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: nlp_data_prepare.py
@time: 2019-05-05 14:56

自然语言处理分为如下几个步骤：
0、先对语料进行规范化
    常见的规范化有：
        去除停用词
        统一大小写
        统一中英文符号
1、word/char 利用语料构建字典
    word_index
    index_word
2、利用字典对语料进行编码/解码
    encode(text):
        :return seq
    decode(seq):
        :return text
3、把语料进行填补至等长 padding

"""
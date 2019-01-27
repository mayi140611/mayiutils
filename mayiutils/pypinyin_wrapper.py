#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: pypinyin_wrapper.py
@time: 2019/1/21 10:59
"""
from pypinyin import Style, pinyin

class PyPinyinWrapper:

    def initialsOfWord(self, word):
        '''
        把汉字词语转换为拼音首字母缩写
        :param word:
        :return: 拼音首字母缩写
        '''
        return ''.join([ii[0] for ii in pinyin(word, style=Style.FIRST_LETTER, strict=False)])
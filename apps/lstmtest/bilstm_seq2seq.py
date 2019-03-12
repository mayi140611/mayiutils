#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: bilstm_seq2seq.py
@time: 2019/3/11 17:02
"""
import re
import numpy as np
import pandas as pd


def clean(s): #整理一下数据，有些不规范的地方
    if '“/s' not in s:
        return s.replace(' ”/s', '')
    elif '”/s' not in s:
        return s.replace('“/s ', '')
    elif '‘/s' not in s:
        return s.replace(' ’/s', '')
    elif '’/s' not in s:
        return s.replace('‘/s ', '')
    else:
        return s


if __name__ == '__main__':
    s = open('msr_train.txt', encoding='gbk').read()
    s = s.split('\r\n')
    # print(s[0])
    s = ''.join(map(clean, s))
    s = re.split('[，。！？、]/[bems]', s)
    print(s[0])















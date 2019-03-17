#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: statistics_wrapper.py
@time: 2019/3/17 9:34

各种统计相关的指标封装
"""
import numpy as np
import pandas as pd
from scipy import stats

if __name__ == '__main__':
    mode = 3
    if mode == 3:
        """
        统计均值mean，中位数median，众数mode
        """
        # arr = np.random.randint(1, 10, [20])
        arr = np.array([1, 9, 6, 4, 8, 5, 7, 3, 2, 8, 3, 8, 3, 9, 7, 9, 5, 1, 5, 3])
        print(np.mean(arr))#5.3
        print(np.median(arr))#5.0
        #np中没有直接求众数的方法
        #bincount统计了每个索引位置出现的次数，注意arr的元素只能是正整数
        #如结果第一个元素0表示0在arr中出现了0次
        print(np.bincount(arr))#[0 2 1 4 1 3 1 2 3 3]
        #间接求众数
        print(np.argmax(np.bincount(arr)))#3
        #也可以使用scipy中的stats直接求众数
        print(stats.mode(arr))#ModeResult(mode=array([3]), count=array([4]))表示3出现了4次
        r = stats.mode(arr)
        print(r[0], r[1])#[3] [4]
        print(r[0][0], r[1][0])#3 4
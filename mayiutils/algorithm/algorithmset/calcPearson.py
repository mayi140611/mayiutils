#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: calcPearson.py
@time: 2019-04-19 15:48

皮尔逊相关系数
https://blog.csdn.net/xzfreewind/article/details/73550856
"""
import numpy as np


def calcPearson(x,y):
    """
    计算x和y两个向量的皮尔逊相关系数
    Pearson系数的取值范围为[-1,1]，当值为负时，为负相关，当值为正时，为正相关，绝对值越大，则正/负相关的程度越大。
    若数据无重复值，且两个变量完全单调相关时，spearman相关系数为+1或-1。当两个变量独立时相关系统为0，但反之不成立
    :param x: 1D ndarray
    :param y: 1D ndarray
    :return:
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    xy = (x - x_mean) * (y - y_mean)
    xy_mean = np.mean(xy)
    p = xy_mean/(np.std(x) * np.std(y))
    return p


if __name__ == '__main__':
    a = np.array([1, 0, 1, 3, 3])
    b = np.array([1, 0.5, 1, 2, 2])
    c = np.array([1, 2, 1, 0, 0])
    print(calcPearson(a, a))
    print(calcPearson(a, b))
    print(calcPearson(a, c))
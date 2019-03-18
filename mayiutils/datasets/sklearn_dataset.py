#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: sklearn_dataset.py
@time: 2019/3/18 10:40
"""
from sklearn import datasets
import numpy as np


if __name__ == "__main__":
    mode = 1
    if mode == 1:
        """
        鸢尾花数据集，三分类
        """
    iris = datasets.load_iris()
    data = iris.data
    y = iris.target
    print(data.shape, y.shape)#(150, 4) (150,)
    print(np.unique(y))#[0 1 2]

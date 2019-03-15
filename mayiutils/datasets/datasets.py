#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: datasets.py
@time: 2019/2/24 9:26

汇总记录用过的数据集及获取方式


1、 CIFAR-10 and CIFAR-100
http://www.cs.toronto.edu/~kriz/cifar.html
from keras.datasets import cifar10  或者cifar100
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train.shape,y_train.shape,x_test.shape,y_test.shape
((50000, 32, 32, 3), (50000, 1), (10000, 32, 32, 3), (10000, 1))
"""


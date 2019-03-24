#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: datasets.py
@time: 2019/2/24 9:26

汇总记录用过的数据集及获取方式
"""
from keras.datasets import imdb
from sklearn import datasets
from keras.datasets import cifar10, cifar100
import numpy as np


if __name__ == "__main__":
    mode = 1
    if mode == 3:
        """
        1、 CIFAR-10 and CIFAR-100 图片10分类或者100分类
        http://www.cs.toronto.edu/~kriz/cifar.html
        """
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        #((50000, 32, 32, 3), (50000, 1), (10000, 32, 32, 3), (10000, 1))

    if mode == 1:
        """
        鸢尾花数据集，三分类
        """
        iris = datasets.load_iris()
        data = iris.data
        y = iris.target
        print(data.shape, y.shape)#(150, 4) (150,)
        print(np.unique(y))#[0 1 2]
    if mode == 2:
        """
        IMDB数据集， 情感二分类
    
        IMDB数据集有5万条来自网络电影数据库的评论；
        其中2万5千条用来训练，2万5千条用来测试，每个部分正负评论各占50%.
        """
        max_features = 2000
        #num_words: max number of words to include
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)


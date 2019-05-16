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
import pandas as pd
import seaborn as sns
import shap  #SHAP package


if __name__ == "__main__":
    mode = 5
    if mode == 5:
        """
        506个样本， 13个特征， 目标值y是自有房屋的价值。可用于做回归
        """
        X, y = shap.datasets.boston()
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        print(type(X), type(y))#<class 'pandas.core.frame.DataFrame'> <class 'numpy.ndarray'>
        print(X.shape, y.shape)  # (506, 13) (506,)
    if mode == 4:
        """
        Diabetes(糖尿病)数据集，可用于做线性回归
        包含442个样本，每个样本有10个特征：
            age, sex, body mass index, average blood pressure, and six blood serum（血清） measurements
        Target: Column 11 is a quantitative（定量） measure of disease progression one year after baseline
        :return:
        """
        data = datasets.load_diabetes()

    if mode == 3:
        """
        1、 CIFAR-10 and CIFAR-100 图片10分类或者100分类
        http://www.cs.toronto.edu/~kriz/cifar.html
        """
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        #((50000, 32, 32, 3), (50000, 1), (10000, 32, 32, 3), (10000, 1))

    if mode == 2:
        """
        IMDB数据集， 情感二分类

        IMDB数据集有5万条来自网络电影数据库的评论；
        其中2万5千条用来训练，2万5千条用来测试，每个部分正负评论各占50%.
        """
        max_features = 2000
        # num_words: max number of words to include
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    if mode == 1:
        """
        鸢尾花数据集，三分类, 150个样本
        有两种方式：
            通过sklearn载入
            通过seaborn载入，直接为df类型
                iris = sns.load_dataset("iris")
        """
        iris = datasets.load_iris()
        # print(iris)
        """
        是一个dict，key有：data, target, target_names, DESCR
        """
        # print(iris.DESCR)
        data = iris.data
        y = iris.target
        print(data.shape, y.shape)#(150, 4) (150,)
        y = iris.target[:, np.newaxis]
        print(y.shape)#(150, 1)
        print(np.unique(y))#[0 1 2]
        dataset = np.concatenate((iris.data, iris.target[:, np.newaxis]), axis=1)
        iris_df = pd.DataFrame(dataset, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class_'])
        print(iris_df.head())
        """
   sepal_length  sepal_width  petal_length  petal_width  class_
0           5.1          3.5           1.4          0.2     0.0
1           4.9          3.0           1.4          0.2     0.0
2           4.7          3.2           1.3          0.2     0.0
3           4.6          3.1           1.5          0.2     0.0
4           5.0          3.6           1.4          0.2     0.0
        """


#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: train.py.py
@time: 2019-05-08 18:01
"""
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from pyod.models.iforest import IForest


if __name__ == '__main__':
    mode = 1
    train_set1 = pd.read_csv('../data/mz_train_data.csv', encoding='gbk', index_col=0)
    train_set = normalize(train_set1.values, axis=0, norm='max')

    # print(train_set.head())
    if mode == 2:
        """
        iForest
        """
        X_train = train_set
        # train IForest detector
        clf_name = 'IForest'
        clf = IForest()
        clf.fit(X_train)

        # get the prediction labels and outlier scores of the training data
        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores
        print(type(y_train_pred))
        s = pd.Series(y_train_pred)
        print(s.value_counts())
        plt.figure()
        plt.boxplot(y_train_scores)
        plt.show()
        s1 = pd.Series(y_train_scores).sort_values(ascending=False)
        print(s1[:100].sort_index())
    if mode == 1:
        """
        Multivariate Gaussian algorithm
        """
        # 1、参数估计
        mu = np.mean(train_set, axis=0)
        sigma = (train_set - mu).T.dot((train_set - mu)) / train_set.shape[0]

        # 2、求出联合概率分布
        def prob(x):
            """
            x是行向量
            :param x:
            :return:
            """
            p = 1 / ((2 * math.pi) ** (train_set.shape[1] * 0.5) * np.linalg.det(sigma) ** 0.5) \
                * math.exp(-0.5 * (x - mu).dot(np.linalg.inv(sigma)).dot((x - mu).T))
            return p

        pp_list = []
        for x in range(train_set.shape[0]):
            pp_list.append(prob(train_set[x]))
        print('正常样本的p值范围：{}-{}'.format(min(pp_list), max(pp_list)))
        plt.figure()
        plt.boxplot(pp_list)
        plt.show()
        s = pd.Series(pp_list)
        print(s[s<0.01].index.shape)
        # # print(s[s>1].shape)
        # s[s<0.01].to_csv('aa.csv', encoding='gbk')
        # s1 = pd.Series(train_set1.index)
        # print(s1[s[s<0.01].index])
        # # t = train_set1[s1[s[s<0.01].index]]
        # # print(t.shape)
        # t = train_set1.iloc[s[s<0.01].index]
        # # print(t.shape)
        # t.to_csv('a.csv', encoding='gbk')

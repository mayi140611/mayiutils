#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: train.py.py
@time: 2019-05-07 17:14
"""
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


if __name__ == '__main__':
    train_set1 = pd.read_csv('zy_train_data.csv', encoding='gbk', index_col=0)
    train_set = normalize(train_set1.values, axis=0, norm='l2')
    # print(train_set.head())
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
    # plt.figure()
    # plt.boxplot(pp_list)
    # plt.show()
    s = pd.Series(pp_list)
    print(s[s<0.01].index)
    # print(s[s>1].shape)
    # s.to_csv('a.csv', encoding='gbk')
    s1 = pd.Series(train_set1.index)
    print(s1[s[s<0.01].index])
    # t = train_set1[s1[s[s<0.01].index]]
    # print(t.shape)
    t = train_set1.iloc[s[s<0.01].index]
    # print(t.shape)
    t.to_csv('a.csv', encoding='gbk')
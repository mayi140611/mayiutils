#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: data_explore.py
@time: 2019/3/27 15:36
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def viewMissingVal(trainDF):
    s = trainDF.isnull().sum() / df.shape[0]
    s_sort = s[s.nonzero()[0]].sort_values(ascending=False)
    return s_sort


if __name__ == "__main__":
    mode = 3
    df = pd.read_csv('anti_fraud_data.csv')
    # print(df.shape)#(46978, 96)
    # print(df.columns)
    traincolumns = df.columns[1: -1]
    labelname = 'flag'
    if mode == 3:
        """
        查看缺失值与非缺失值对欺诈的影响（缺失值对应的欺诈样本概率和非欺诈对应的欺诈样本概率）
        """
        s_sort = viewMissingVal(df[traincolumns])
        logoddsdict = {}
        for i in s_sort.index:
            if s_sort[i] == 1:
                print('{} 完全缺失'.format(i))
                continue
            tempDF = df.loc[:, [i, labelname]]
            s1 = tempDF[tempDF[i].isnull()][labelname].value_counts()
            s2 = tempDF[tempDF[i].notnull()][labelname].value_counts()
            rate1 = s1[1]/(s1[0]+s1[1])
            rate2 = s2[1]/(s2[0]+s2[1])
            print('{} 缺失对应的欺诈率:{:.4f}% 非缺失对应的欺诈率:{:.4f}%'.format(i, rate1*100, rate2*100))
            logoddsdict[i] = np.log10(rate1/rate2)
        s1 = pd.Series(logoddsdict).sort_values(ascending=False)
        print(s1.index)
        #当log odds显著地不等亍0，说明缺失对亍欺诈是存在一定的影响的。
        plt.figure()
        plt.bar(list(range(len(s1))), s1.values)
        # plt.bar(list(s1.index), s1.values)
        plt.show()
    if mode == 2:
        """
        查看样本的缺失值
        """
        # 获取缺失率
        s_sort = viewMissingVal(df[traincolumns])
        print(s_sort)
        plt.figure()
        plt.bar(list(range(len(s_sort))), s_sort.values)
        plt.show()


    if mode == 1:
        """
        查看label的分布
        """

        print(df[labelname].value_counts())
        """
        0    46457
        1      521
        Name: flag, dtype: int64       
        """
#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: data_exploration.py
@time: 2019/3/1 18:18
"""
import pandas as pd
import matplotlib.pyplot as plt


from mayiutils.algorithm.dataprepare import DataPrepare as dp
if __name__ == '__main__':
    traindf = pd.read_csv('D:/Desktop/DF/portrait/train_dataset.csv')
    print(traindf.values[:5, 29:])
    print(traindf.values[:5, 29])
    # print(traindf.shape)#(50000, 30)
    testdf = pd.read_csv('D:/Desktop/DF/portrait/test_dataset.csv')
    # print(testdf.shape)#(50000, 29)
    # print(traindf.info())#可以反映每列是否有空值，每列类型
    s = dp.checkTarget(traindf, '信用分')
    # s = dp.checkTarget(traindf, '信用分', mode='regression')
    """
    取值范围: 422-719
    平均数: 618.05306
    中位数: 627.0
    %25-75%: 594.0-649.0
    %5-95%: 535.0-673.0
    %1-99%: 497.0-688.0
    """
    plt.figure()
    plt.hist(traindf['信用分'].values, 30)
    # plt.subplot(2, 1, 1)
    # plt.boxplot(traindf['信用分'])
    # plt.subplot(2, 1, 2)
    # s.sort_index().hist()
    plt.show()
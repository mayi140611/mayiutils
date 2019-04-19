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


from mayiutils.datasets.dataprepare import DataPrepare as dp
if __name__ == '__main__':
    mode = 2
    traindf = pd.read_csv('D:/Desktop/DF/portrait/train_dataset.csv')
    # print(traindf.values[:5, 29:])
    # print(traindf.values[:5, 29])
    if mode == 2:
        """
        增加目标值集中程度，把小于470的都替换成470
        年龄有0岁的，替换成中位数，有超过100岁的，替换成90岁
        
        使用GBDT的话好像对异常值不敏感吧，而且分裂点他也会自己算。。。
        用户网龄（月）：明显的网龄越长，信用分越高
        """
        print(traindf['用户年龄'].max(), traindf['用户年龄'].min())
        # 观察用户年龄和目标值的分布
        # print(traindf[['用户年龄', '信用分']].value_counts())
        # plt.figure()
        # plt.scatter(traindf['用户年龄'].values, traindf['信用分'].values)
        # plt.scatter(traindf['用户网龄（月）'].values, traindf['信用分'].values, alpha=0.5)
        # plt.scatter(traindf['用户最近一次缴费距今时长（月）'].values, traindf['信用分'].values, alpha=0.5)
        # plt.scatter(traindf['缴费用户最近一次缴费金额（元）'].values, traindf['信用分'].values, alpha=0.5)
        # plt.scatter(traindf['用户近6个月平均消费值（元）'].values, traindf['信用分'].values, alpha=0.5)
        # plt.show()
        print(traindf.loc[:, '用户年龄'].median())
        traindf.loc[traindf['信用分'] < 470, '信用分'] = 470
        traindf.loc[traindf['用户年龄'] == 0, '用户年龄'] = traindf.loc[:, '用户年龄'].median()
        traindf.loc[traindf['用户年龄'] > 90, '用户年龄'] = 90
        traindf.to_csv('D:/Desktop/DF/portrait/train_datasetp.csv', encoding='utf8', index=False)
        print(traindf['用户年龄'].max(), traindf['用户年龄'].min())
    if mode == 1:
        """
        查看目标值分布
        """

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
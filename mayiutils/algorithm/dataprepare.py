#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: dataprepare.py
@time: 2019/3/2 7:12
"""
import numpy as np


class DataPrepare:
    """
    包含traindata和testdata的数据探索和预处理，返回可以直接训练的traindataset， valdataset，testdataset
    第一个阶段：数据探索
        1、查看目标值的取值情况，如果是分类问题，绘制频率直方图；如果是回归问题，绘制出分布、箱线图等

    """
    @classmethod
    def checkTarget(cls, df, targetColoum, mode='classification'):
        if mode == 'classification':
            s = df[targetColoum].value_counts().sort_values(ascending=False)
            print('样本总数：{}'.format(df.shape[0]))
            print('类别数：{}'.format(s.shape[0]))
            print(s.sort_index())
            return s
        elif mode == 'regression':
            ## 最大值，最小值
            # 转换为numpy格式
            arr = df[targetColoum].values
            max = np.max(arr)
            min = np.min(arr)
            mean = np.mean(arr)
            median = np.median(arr)
            p5 = np.percentile(arr, 5)
            p95 = np.percentile(arr, 95)
            p1 = np.percentile(arr, 1)
            p99 = np.percentile(arr, 99)
            p25 = np.percentile(arr, 25)
            p75 = np.percentile(arr, 75)
            print('取值范围: {}-{}'.format(min, max))
            print('平均数: {}'.format(mean))
            print('中位数: {}'.format(median))
            print('%25-75%: {}-{}'.format(p25, p75))
            print('%5-95%: {}-{}'.format(p5, p95))
            print('%1-99%: {}-{}'.format(p1, p99))




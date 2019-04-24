#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: data_prepare.py
@time: 2019/3/2 7:12
"""
import numpy as np


class DataPrepare:
    """
    依次做以下几件事情：
    1、数据探索 data_explore
        label
            数据的label位于哪一列，
            离散值：分类问题；
                几分类
                每个分类的样本数量
                    样本是否均匀
            连续值：回归问题
                分布
                最大值最小值
        features
            每个feature的类型：
                分类变量？
                    几分类
                    value_counts()
                    如何编码？
                        如果类别之间有顺序关系，如 危险等级 1-7级，就直接编码为数值的1-7
                        如果类别之间没有顺序关系，如 男女，用one-hot
                数值型？
                时间类型？
                    在第2步 features-engineering中处理
            每个feature是否存在缺失？缺失率？缺失值如何处理？
                如果缺失率大于一定的阈值，如0.99，直接删掉该特征
                如果有少量缺失，视具体情况处理
                    可以把缺失单独看做一种状态；
                    或者按某种规则填充
            异常值处理
            归一化？
                由于现在基本上就用GBDT等tree模型了，所以不用做特征归一化！！！
    2、features-engineering
        features衍生
            用feature tools
        features reduction
            features-selector
    3、数据集拆分
        traindataset
            在训练的时候
            第一步，使用grid通过交叉验证cv选择最优参数
            第二步，再用最优参数在整个训练集上train得到最终的模型
        testdataset

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




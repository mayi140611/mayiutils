#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: data_preprocessing.py
@time: 2019/3/2 7:12

数据预处理分为3个阶段：
    1、data explore
        了解初始数据
            label在哪个字段，
            label的类型，
                连续值：分布
                离散值：类别数量、分布是否均衡
            有哪些字段，
            每个字段的业务含义，
            每个字段的类型：分类变量 or 数值变量
        制定目标：分类问题 / 回归问题 / 无监督问题
        数据处理：features selector
            删除掉完全缺失的字段和唯一值字段
            对缺失率很高的字段进行处理
                获取缺失字段对label的影响
            异常值处理
            类别变量转换
                如果有顺序关系，就用数值表示
                如果没有顺序关系，则用one-hot
            删除相关性较高的变量
    2、data feature engineering 特征工程
        2.1 特征衍生 Feature derivation
            featuretools
            根据业务关系衍生
        2.2 特征筛选 features selector
            删除相关性较高的变量
            删除重要性低的变量
    3、model
        建立模型
    4、train.py  & train & eval & predict
        模型训练，架子是一样的，模型可插拔 Pluggable
    5、main
        data prepare
            根据模型要求决定数据是否归一化
                由于现在基本上就用GBDT等tree模型了，所以不用做特征归一化！！！
                如果不是用tree，则需要归一化
                    https://blog.csdn.net/lyhope9/article/details/82778459
            train_set val_set test_set splitter

"""
import numpy as np
import pandas as pd
# from sklearn.preprocessing import normalize


class DataExplore:
    @classmethod
    def calMissRate(cls, df, col):
        """
        计算某一列的缺失率
        :param df:
        :param col:
        :return:
        """
        r = df[df[col].isnull()].shape[0] / df.shape[0]
        print(f'字段{col}的缺失率为：{round(r, 2)}')
        return r

    @classmethod
    def calMissRateImpact2Labels(cls, df, col, labelCol, labels, verbose=True):
        """
        计算缺失率对2分类的影响
        :param df:
        :param col:
        :param labelCol:
        :param labels: list. [label1, label2]
        :return:
        """
        df_null = df[df[col].isnull()]
        a = df_null[labelCol].value_counts()
        if verbose:
            print(a)
        if 0 in a:
            r1 = 1.0 * a[labels[0]] / df_null.shape[0]
        else:
            r1 = 0
        print(f'特征 {col} 缺失时, label {labels[0]} 占总样本数比 = {round(r1, 2)}')
        df_notnull = df[df[col].notnull()]
        a = df_notnull[labelCol].value_counts()
        if verbose:
            print(a)
        if 0 in a:
            r2 = 1.0 * a[labels[0]] / df_notnull.shape[0]
        else:
            r2 = 0
        print(f'特征 {col} 非缺失时, label {labels[0]} 占总样本数比 = {round(r2, 2)}')
        return r1, r2


    @classmethod
    def build_one_hot_features(cls, df, cols):
        """
        构建one-hot特征
        :param df:
        :param cols: list
        :return:
        """
        for col in cols:
            t = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, t], axis=1)
            del df[col]
        return df

    @classmethod
    def hosRank(cls, hosRankSeries):
        """
        把医院的等级描述转化为相应的数字
        :param hosRankSeries: 医院等级的Series
        :return:
        """
        def t(x):
            if x.find('三级') != -1:
                return 3
            if x.find('二级') != -1:
                return 2
            if x.find('一级') != -1:
                return 1
            if x.find('未评级') != -1:
                return 0
        return hosRankSeries.apply(t)


class DataPrepare:
    """

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


if __name__ == '__main__':
    mode = 1
    if mode == 1:
        """
        data explore
        """
        pass
        # # 删除列全为空的字段
        # mzdf4 = mzdf4.dropna(axis=1, how='all')
        # # 删除行全为空的字段
        # mzdf4 = mzdf4.dropna(axis=0, how='all')

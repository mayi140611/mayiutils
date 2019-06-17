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
from sklearn.preprocessing import normalize


class DataExplore:
    @classmethod
    def describe(cls, df):
        """
        描述df的
            data types
            percent missing
            unique values
            mode 众数
            count mode 众数计数
            % mode 众数占所有数据的百分比
            distribution stats  分布数据 分位数
        :param df:
        :return:
        """
        # data types
        dqr_data_types = pd.DataFrame(df.dtypes, columns=['Data Type'])

        # percent missing
        dqr_percent_missing = pd.DataFrame(100 * (df.isnull().sum() / len(df)).round(3), columns=['% Missing'])

        # unique values
        dqr_unique_values = pd.DataFrame(columns=['Unique Values'])
        for c in df:
            dqr_unique_values.loc[c] = df[c].nunique()

        # mode 众数
        dqr_mode = pd.DataFrame(df.mode().loc[0])
        dqr_mode.rename(columns={dqr_mode.columns[0]: "Mode"}, inplace=True)

        # count mode
        dqr_count_mode = pd.DataFrame(columns=['Count Mode'])
        for c in df:
            dqr_count_mode.loc[c] = df[c][df[c] == dqr_mode.loc[[c]].iloc[0]['Mode']].count()

            # % mode
        dqr_percent_mode = pd.DataFrame(100 * (dqr_count_mode['Count Mode'].values / len(df)), \
                                        index=dqr_count_mode.index, columns=['% Mode'])

        # distribution stats
        df['temp_1a2b3c__'] = 1
        dqr_stats = pd.DataFrame(df['temp_1a2b3c__'].describe())
        del df['temp_1a2b3c__']
        for c in df:
            dqr_stats = dqr_stats.join(pd.DataFrame(df[c].describe()))
        del dqr_stats['temp_1a2b3c__']
        dqr_stats = dqr_stats.transpose().drop('count', axis=1)

        print("num of records: {}, num of columns: {}".format(len(df), len(df.columns)))

        return dqr_data_types.join(dqr_unique_values[['Unique Values']].astype(int)). \
            join(dqr_percent_missing).join(dqr_mode).join(dqr_count_mode[['Count Mode']].astype(int)).join(dqr_percent_mode).join(dqr_stats)

    @classmethod
    def normalize(cls, X, norm='l2', axis=0):
        """
        归一化
        :param X:
        :param norm:'l1', 'l2', or 'max', optional ('l2' by default)
            The norm to use to normalize each non zero sample (or each non-zero
            feature if axis is 0).
        :param axis:0 or 1, optional (1 by default)
            axis used to normalize the data along. If 1, independently normalize
            each sample, otherwise (if 0) normalize each feature.
        :return:
        """
        return normalize(X, norm, axis)

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
    pass


if __name__ == '__main__':
    mode = 1
    if mode == 1:
        """
        data explore
        """
        x = np.arange(9).reshape((3, 3))
        print(x)
        """
[[0 1 2]
 [3 4 5]
 [6 7 8]]
        """
        print(DataExplore.normalize(x, norm='max'))
        """
[[0.         0.14285714 0.25      ]
 [0.5        0.57142857 0.625     ]
 [1.         1.         1.        ]]
        """
        print(DataExplore.normalize(x, norm='l1'))
        """
[[0.         0.08333333 0.13333333]
 [0.33333333 0.33333333 0.33333333]
 [0.66666667 0.58333333 0.53333333]]
        """
        print(DataExplore.normalize(x, norm='l2'))
        """
[[0.         0.12309149 0.20739034]
 [0.4472136  0.49236596 0.51847585]
 [0.89442719 0.86164044 0.82956136]]
        """
        # # 删除列全为空的字段
        # mzdf4 = mzdf4.dropna(axis=1, how='all')
        # # 删除行全为空的字段
        # mzdf4 = mzdf4.dropna(axis=0, how='all')

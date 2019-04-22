#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: feature_selector_wrapper.py
@time: 2019-04-22 18:41

https://github.com/WillKoehrsen/feature-selector

从5个维度削减特征
There are five methods used to identify features to remove:

Missing Values
Single Unique Values
Collinear Features
Zero Importance Features
Low Importance Features



使用：
拿到数据后，
1、要把日期数据处理掉，如转换为天数等数值变量；
2、把与业务无关的变量删掉，如客户编号等；
再用feature-selector处理
"""
import pandas as pd
import numpy as np
from feature_selector import FeatureSelector


if __name__ == '__main__':
    mode = 1
    df = pd.read_excel('/Users/luoyonggui/Documents/work/dataset/0/data.xlsx')
    # print(df.info())
    """
RangeIndex: 562 entries, 0 to 561
Data columns (total 42 columns):
Unnamed: 0                              562 non-null int64
平台流水号                                   562 non-null int64
保单管理机构                                  562 non-null int64
保单号                                     562 non-null int64
指定受益人标识                                 562 non-null object
受益人与被保险人关系                              538 non-null object
交费方式                                    562 non-null object
交费期限                                    562 non-null int64
核保标识                                    562 non-null object
核保结论                                    544 non-null object
投保时年龄                                   562 non-null int64
基本保额与体检保额起点比例                           0 non-null float64
生调保额起点                                  0 non-null float64
投保保额临近核保体检临界点标识                         0 non-null float64
投保保额                                    562 non-null float64
临近核保生调临界点标识                             0 non-null float64
理赔金额                                    562 non-null float64
累计已交保费                                  562 non-null float64
理赔结论                                    562 non-null object
Unnamed: 19                             562 non-null int64
生效日期                                    562 non-null datetime64[ns]
出险前最后一次复效日期                             6 non-null datetime64[ns]
承保后最小借款日期                               2 non-null datetime64[ns]
出险日期                                    562 non-null datetime64[ns]
报案时间                                    119 non-null datetime64[ns]
申请日期                                    562 non-null datetime64[ns]
出险减生效天数                                 562 non-null int64
出险减最后一次复效天数                             6 non-null float64
重疾保单借款减生效日期天数                           0 non-null float64
申请时间减出险时间                               562 non-null int64
报案时间减出险时间                               119 non-null float64
出险原因1                                   562 non-null object
出险原因2                                   0 non-null float64
出险原因3                                   0 non-null float64
出险结果                                    552 non-null object
保单借款展期未还次数                              0 non-null float64
失复效记录次数                                 562 non-null int64
销售渠道                                    562 non-null object
(SELECTDISTINCTLJ.AGENTCODEFRO销售人员工号    562 non-null int64
被保人核心客户号                                562 non-null int64
保人归并客户号                                 562 non-null int64
被保人归并客户号                                562 non-null int64
dtypes: datetime64[ns](6), float64(13), int64(14), object(9)
    """
    #提取出标签数据
    label = df['理赔结论']
    label[label != '正常给付'] = int(1)
    label[label == '正常给付'] = int(0)
    label = np.array(list(label))
    df = df.drop(columns=['理赔结论'])

    df = df.drop(columns=['Unnamed: 0', '平台流水号', 'Unnamed: 19', '生效日期', '出险日期', '报案时间', '申请日期', '出险前最后一次复效日期', '承保后最小借款日期'])

    if mode == 0:
        """
        标准的data explore步骤
        """

        # print(df.info())# 查看df字段和缺失值信息



        # train_col.remove('平台流水号')
        # train_col.remove('Unnamed: 19')
        # train_col.remove('生效日期')
        # train_col.remove('出险日期')
        # train_col.remove('报案时间')
        # train_col.remove('申请日期')
        # train_col.remove('出险减最后一次复效天数')
        # train_col.remove('报案时间减出险时间')
        # train_col.remove('出险前最后一次复效日期')
        # train_col.remove('承保后最小借款日期')

        fs = FeatureSelector(data=df, labels=label)
        # 缺失值处理
        """
        查找缺失率大于0.6的特征
        """
        fs.identify_missing(missing_threshold=0.6)
        """
        13 features with greater than 0.60 missing values.
        """
        missing_features = fs.ops['missing']
        # 查看缺失特征
        print(missing_features[:10])
        """
        ['基本保额与体检保额起点比例', '生调保额起点', '投保保额临近核保体检临界点标识', '临近核保生调临界点标识', '出险前最后一次复效日期', '承保后最小借款日期', '报案时间', '出险减最后一次复效天数', '重疾保单借款减生效日期天数', '报案时间减出险时间']
        """
        # fs.plot_missing()
        # 查看每个特征的缺失率
        print(fs.missing_stats)

        # 单一值
        fs.identify_single_unique()
        """
        0 features with a single unique value.
        """
        single_unique = fs.ops['single_unique']
        print(single_unique)
        # fs.plot_unique()
        # Collinear (highly correlated) Features
        fs.identify_collinear(correlation_threshold=0.975)
        """
        2 features with a correlation magnitude greater than 0.97.
        """
        correlated_features = fs.ops['collinear']
        print(correlated_features[:5])
        """
        ['报案时间减出险时间', '被保人归并客户号']
        """
        # fs.plot_collinear()
        # fs.plot_collinear(plot_all=True)
        print(fs.record_collinear.head())
        """
          drop_feature corr_feature  corr_value
0    报案时间减出险时间    申请时间减出险时间    0.985089
1     被保人归并客户号     被保人核心客户号    1.000000
        """
        # 4. Zero Importance Features
        fs.identify_zero_importance(task='classification', eval_metric='auc',
                                    n_iterations=10, early_stopping=True)
        one_hot_features = fs.one_hot_features
        base_features = fs.base_features
        print('There are %d original features' % len(base_features))
        print('There are %d one-hot features' % len(one_hot_features))
        """
        There are 33 original features
There are 212 one-hot features
        """
        print(fs.one_hot_features[:20])
        # print(fs.data_all.head(10))
        print(fs.data_all.shape)

        zero_importance_features = fs.ops['zero_importance']
        print(zero_importance_features[:5])
        # fs.plot_feature_importances(threshold=0.99, plot_n=12)
        print(fs.feature_importances.head(10))
#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: data_explore_example.py
@time: 2019-05-17 16:02

参考样例：https://mp.weixin.qq.com/s/hb8dlm3Ixsn9vcDfInqAZQ
"""
import pandas as pd
pd.set_option('display.max_columns', 100)  # 设置显示数据的最大列数，防止出现省略号…，导致数据显示不全
pd.set_option('expand_frame_repr', False)  # 当列太多时不自动换行
from mayiutils.datasets.data_preprocessing import DataExplore as de
from feature_selector import FeatureSelector

if __name__ == '__main__':
    mzdf = pd.read_csv('/Users/luoyonggui/PycharmProjects/mayiutils/aawork/proj3/data/mz_all.csv', index_col=0, encoding='gbk', parse_dates=['生效日期', '出生日期', '就诊结帐费用发生日期']\
                       , converters={'出险人客户号': str, '主被保险人客户号': str}, nrows=1000)
    """
    【header】默认header=0，即将文件中的0行作为列名和数据的开头，但有时候0行的数据是无关的，我们想跳过0行，让1行作为数据的开头，可以通过将header设置为1来实现。
    【usecols】根据列的位置或名字，如[0,1,2]或[‘a’, ‘b’, ‘c’]，选出特定的列。
    index_col=0 : 读取第0列作为索引
    parse_dates=['生效日期', '出生日期', '就诊结帐费用发生日期'] ： 把相应列解析为datetime类型
    converters={'出险人客户号': str, '主被保险人客户号': str}： 把相应列以string类型读取。这个主要应用于一些客户号读取时是按照int读取的，那么前面的0会被去掉
    nrows=1000 : 只读取前1000行
    """
    print(mzdf.sample(2))
    mzdf_info = de.describe(mzdf)
    print(mzdf_info)
    print(mzdf.describe())
    # # 删除完全缺失字段
    mzdf.dropna(axis=1, how='all', inplace=True)
    # 删除指定列
    to_drop = ['Job #', 'Doc #']
    mzdf.drop(to_drop, axis=1, inplace=True)
    #
    fs = FeatureSelector(data=mzdf)
    fs.identify_missing(missing_threshold=0.1)  # 找出缺失率大于0.1的特征
    print(fs.record_missing)
    fs.identify_single_unique()
    print(fs.record_single_unique)
    fs.identify_collinear(correlation_threshold=0.975)
    print(fs.record_collinear)
    mzdf = fs.remove(methods=['missing', 'single_unique', 'collinear'])
    # 重命名字段
    new_names = {'Borough': '区', 'Initial Cost': '初始成本', 'Total Est. Fee': '总附加费用'}
    mzdf.rename(columns=new_names, inplace=True)

    # 设置新索引 会把某字段设置成索引列，通过该字段从字段列移除
    mzdf.set_index('区', inplace=True)

    mzdf['医院等级'] = de.hosRank(mzdf['医院等级'])


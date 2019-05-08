#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: data_explore.py
@time: 2019-05-08 10:21
"""
import pandas as pd
from feature_selector import FeatureSelector
from mayiutils.datasets.data_preprocessing import DataExplore as de

if __name__ == '__main__':
    mode = 2
    if mode == 2:
        """
        
        """
        mzdf = pd.read_csv('../data/mz_all.csv', encoding='gbk', index_col=0, parse_dates=['生效日期', '出生日期', '就诊结帐费用发生日期'])
        # print(len(list(mzdf.index)))#996271 总案号没有重复
        del mzdf['ROWNUM']
        # 删除名称 因为有代码
        del mzdf['主被保险人']
        del mzdf['出险人姓名']
        del mzdf['险种名称']
        del mzdf['医院名称']
        del mzdf['费用项目代码']
        del mzdf['疾病名称']
        # print(mzdf['保单号'].value_counts())
        # print(mzdf['生效日期'].value_counts())
        # print(mzdf['人员属性'].value_counts())
        # print(mzdf['证件类型'].value_counts())
        # print(mzdf['性别'].value_counts())
        # print(mzdf['出险原因'].value_counts())
        # print(mzdf['险种代码'].value_counts())
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='门诊疾病就诊'] = '门诊就诊'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='牙科医疗'] = '牙科治疗'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='牙齿护理'] = '牙科治疗'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='牙科护理'] = '牙科治疗'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='牙科保健'] = '牙科治疗'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='紧急牙科治疗'] = '牙科治疗'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='生育'] = '普通生育门诊'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='其他约定1门诊'] = '其他约定'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='其他约定1'] = '其他约定'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='统筹约定'] = '统筹门诊'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='住院'] = '统筹门诊'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='门诊意外首次就诊'] = '门诊意外就诊'
        # print(mzdf['就诊类型名称'].value_counts())
        mzdf['费用项目名称'][mzdf['费用项目名称']=='诊疗费'] = '治疗费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='门诊手术费'] = '手术费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='CT费'] = 'MRI/CT费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='核磁费'] = 'MRI/CT费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='放射费'] = 'MRI/CT费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='中成药费'] = '中成药'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='专家挂号费'] = '挂号费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='合计金额'] = '其他费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='氧气费'] = '输氧费'
        print(mzdf['费用项目名称'].value_counts())
        # print(mzdf['疾病代码'].value_counts())

        mzdf = de.build_one_hot_features(mzdf, ['保单号', '生效日期', '人员属性', '证件类型', '性别', '出险原因', '险种代码', '就诊类型名称', '费用项目名称'])
        mzdf.info()
        # train_removedg = train_removed.groupby(['总案号', '收据号'])[['费用金额', '自费金额', '部分自付金额', '医保支付金额']].sum()
        # train_removed = train_removed.drop_duplicates(subset=['总案号', '收据号'], keep='first')
        # print(train_removed.shape)#(644577, 31)
        # del train_removed['费用金额']
        # del train_removed['自费金额']
        # del train_removed['部分自付金额']
        # del train_removed['医保支付金额']
        # train_removed = pd.merge(train_removed, train_removedg, how='left', left_on=['总案号', '收据号'], right_index=True)


    if mode == 1:
        """
        明细数据
        """
        mzdf4 = pd.read_csv('../data/mz4.csv').iloc[:, 2:]
        mzdf5 = pd.read_csv('../data/mz5.csv').iloc[:, 2:]
        mzdf6 = pd.read_csv('../data/mz6.csv').iloc[:, 2:]
        mzdf = pd.concat([mzdf4, mzdf5, mzdf6])
        fs = FeatureSelector(data=mzdf)
        fs.identify_missing(missing_threshold=0.6)
        # fs.plot_missing()
        # 查看每个特征的缺失率
        # print(fs.missing_stats)
        # 查看超过阈值的特征
        print(fs.record_missing)
        """
                              feature  missing_fraction
    0  (SELECTDISTINCT(P.PAYENDDATE-1          1.000000
    1                            身份标识          1.000000
    2                            身份备注          1.000000
    3                            证件号码          1.000000
    4                            发票类型          1.000000
    5                            自费描述          0.753404
    6                          部分自付描述          0.907056
    7                          医保支付描述          0.986467
    8                            入院时间          1.000000
    9                            出院时间          1.000000
        """
        fs.identify_single_unique()
        print(fs.record_single_unique)
        """
      feature  nunique
    0  投保单位名称        1
    1    就诊类型        1
        """
        fs.identify_collinear(correlation_threshold=0.975)
        print(fs.record_collinear)
        """
      drop_feature corr_feature  corr_value
    0          分案号          总案号         1.0
        """
        train_removed= fs.remove(methods=['missing', 'single_unique', 'collinear'])
        # train_removed.info()
        """
Int64Index: 996271 entries, 2017221870455 to 2019222749669
Data columns (total 30 columns):
ROWNUM        996271 non-null int64
保单号           996271 non-null int64
生效日期          996271 non-null object
出险人姓名         996271 non-null object
主被保险人         996271 non-null object
主被保险人客户号      996271 non-null int64
人员属性          996271 non-null object
证件类型          996271 non-null int64
出险人客户号        996271 non-null int64
性别            996271 non-null object
年龄            996271 non-null int64
出生日期          996271 non-null object
出险原因          996271 non-null object
险种名称          996271 non-null object
险种代码          996271 non-null object
收据号           996271 non-null object
医院代码          996271 non-null object
医院名称          996271 non-null object
医院等级          996271 non-null object
就诊结帐费用发生日期    996271 non-null object
费用合计          996271 non-null float64
费用项目代码        996271 non-null object
费用项目名称        996271 non-null object
费用金额          996271 non-null float64
疾病代码          996271 non-null object
疾病名称          996271 non-null object
就诊类型名称        995493 non-null object
自费金额          996271 non-null float64
部分自付金额        996271 non-null float64
医保支付金额        996271 non-null float64
dtypes: float64(5), int64(6), object(19)
memory usage: 235.6+ MB
        """
        train_removed.index = train_removed['总案号']
        del train_removed['总案号']
        train_removed['医院等级'] = de.hosRank(train_removed['医院等级'])
        train_removed.info()
        train_removed.to_csv('../data/mz_all.csv', encoding='gbk', index=True)
        train_removed.iloc[:10000].to_csv('../data/mz_all1.csv', encoding='gbk', index=True)
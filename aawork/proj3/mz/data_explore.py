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
import math


if __name__ == '__main__':
    mode = 5
    if mode == 5:
        """
        特征工程
        """
        mzdf = pd.read_csv('../data/mz_all_claim.csv', encoding='gbk', index_col=0,
                           parse_dates=['出生日期', '就诊结帐费用发生日期'])
        mzdf['出生月'] = mzdf['出生日期'].dt.month
        mzdf['就诊结帐费用发生月'] = mzdf['就诊结帐费用发生日期'].dt.month
        mzdf['就诊结帐费用发生月'] = mzdf['就诊结帐费用发生日期'].dt.weekday
        del mzdf['主被保险人客户号']
        del mzdf['出险人客户号']
        del mzdf['出生日期']
        del mzdf['收据号']
        del mzdf['医院代码']
        del mzdf['就诊结帐费用发生日期']
        del mzdf['费用合计']
        del mzdf['疾病代码']
        mzdf.index = mzdf['总案号_分案号']
        del mzdf['总案号_分案号']
        del mzdf['费用金额']
        del mzdf['自费金额']
        del mzdf['部分自付金额']
        del mzdf['医保支付金额']
        del mzdf['自费总金额']
        del mzdf['费用金额mean']

        fs = FeatureSelector(data=mzdf)
        fs.identify_missing(missing_threshold=0.6)
        print(fs.record_missing)

        fs.identify_single_unique()
        print(fs.record_single_unique)
        fs.identify_collinear(correlation_threshold=0.975)
        print(fs.record_collinear)
        """
        """
        train_removed = fs.remove(methods=['missing', 'single_unique', 'collinear'])
        train_removed.info()
        # train_removed.to_csv('../data/mz_train_data.csv', encoding='gbk', index=True)

    if mode == 4:
        """
        事件压成赔案
            把费用和相对应的就诊类型名称相加
            统计每个赔案包含的事件数
            统计每个赔案的去医院数量
            统计每个赔案的诊断数量
            统计每个赔案的出险人数量
            统计每个赔案去的不同等级的医院数量 暂时没做！
            各种费用占总费用比
            各种费用的log（去掉原费用）
        """
        mzdf = pd.read_csv('../data/mz_all_event.csv', encoding='gbk', index_col=0,
                           parse_dates=['出生日期', '就诊结帐费用发生日期'])
        flist = [
            '费用金额', '自费金额', '部分自付金额', '医保支付金额',
            '费用项目名称_中成药费', '费用项目名称_中草药', '费用项目名称_其他费', '费用项目名称_化验费',
            '费用项目名称_床位费', '费用项目名称_手术费', '费用项目名称_护理费', '费用项目名称_挂号费',
            '费用项目名称_材料费', '费用项目名称_检查费', '费用项目名称_治疗费', '费用项目名称_西药费',
            '费用项目名称_诊疗费'
        ]
        train_removedg = mzdf.groupby(['总案号_分案号'])[flist].sum()
        train_removedg1 = mzdf.groupby(['总案号_分案号'])['收据号'].count()
        train_removedg1.columns = ['event_num']
        train_removedg1.name = 'event_num'
        print(train_removedg1.columns)
        print(train_removedg1.head())
        # 统计去不同医院的数量
        def t(arr):
            return arr.unique().shape[0]
        train_removedg2 = mzdf.groupby(['总案号_分案号'])['医院等级'].agg(t)
        train_removedg2.columns = ['hos_num']
        train_removedg2.name = 'hos_num'
        # 统计诊断的数量
        train_removedg3 = mzdf.groupby(['总案号_分案号'])['疾病代码'].agg(t)
        train_removedg3.columns = ['疾病代码_num']
        train_removedg3.name = '疾病代码_num'
        # 统计出险人的数量
        train_removedg4 = mzdf.groupby(['总案号_分案号'])['出险人客户号'].agg(t)
        train_removedg4.columns = ['出险人客户号_num']
        train_removedg4.name = '出险人客户号_num'

        mzdf = mzdf.drop_duplicates(subset=['总案号_分案号'], keep='first')
        for i in flist:
            del mzdf[i]
        mzdf = pd.merge(mzdf, train_removedg, how='left', left_on=['总案号_分案号'], right_index=True)
        mzdf = pd.merge(mzdf, train_removedg1, how='left', left_on=['总案号_分案号'], right_index=True)
        mzdf = pd.merge(mzdf, train_removedg2, how='left', left_on=['总案号_分案号'], right_index=True)
        mzdf = pd.merge(mzdf, train_removedg3, how='left', left_on=['总案号_分案号'], right_index=True)
        mzdf = pd.merge(mzdf, train_removedg4, how='left', left_on=['总案号_分案号'], right_index=True)

        # 构造特征
        # 自费金额占比
        mzdf['自费总金额'] = mzdf['自费金额'] + mzdf['部分自付金额']
        # 自费总金额占费用金额比
        mzdf['自费总金额占比'] = mzdf['自费总金额'] / mzdf['费用金额']
        # 医保支付金额占比
        mzdf['医保支付金额占比'] = mzdf['医保支付金额'] / mzdf['费用金额']
        # 平均每次事件费用金额
        mzdf['费用金额mean'] = mzdf['费用金额'] / mzdf['event_num']


        # log
        def tlog(x):
            if x < 1:
                x = 0
            if x != 0:
                x = math.log10(x)
            return x
        mzdf['费用金额log'] = mzdf['费用金额'].apply(tlog)
        mzdf['自费金额log'] = mzdf['自费金额'].apply(tlog)
        mzdf['部分自付金额log'] = mzdf['部分自付金额'].apply(tlog)
        mzdf['医保支付金额log'] = mzdf['医保支付金额'].apply(tlog)
        mzdf['自费总金额log'] = mzdf['自费总金额'].apply(tlog)
        mzdf['费用金额meanlog'] = mzdf['费用金额mean'].apply(tlog)
        print(mzdf.shape)  # (267303, 74)
        mzdf.info()
        """
Int64Index: 152420 entries, 0 to 301570
Data columns (total 74 columns):
主被保险人客户号                    152420 non-null int64
出险人客户号                      152420 non-null int64
年龄                          152420 non-null int64
出生日期                        152420 non-null datetime64[ns]
收据号                         152420 non-null object
医院代码                        152420 non-null object
医院等级                        152420 non-null int64
就诊结帐费用发生日期                  152420 non-null datetime64[ns]
费用合计                        152420 non-null float64
疾病代码                        152420 non-null object
总案号_分案号                     152420 non-null object
保单号_82200946504             152420 non-null int64
保单号_82200946505             152420 non-null int64
保单号_82200946506             152420 non-null int64
生效日期_2016-06-01 00:00:00    152420 non-null int64
生效日期_2017-06-01 00:00:00    152420 non-null int64
生效日期_2018-06-01 00:00:00    152420 non-null int64
人员属性_主被保险人                  152420 non-null int64
人员属性_连带被保险人                 152420 non-null int64
证件类型_0                      152420 non-null int64
证件类型_1                      152420 non-null int64
证件类型_8                      152420 non-null int64
性别_女                        152420 non-null int64
性别_男                        152420 non-null int64
出险原因_意外                     152420 non-null int64
出险原因_疾病                     152420 non-null int64
险种代码_MIK01                  152420 non-null int64
险种代码_NIK01                  152420 non-null int64
险种代码_NIK02                  152420 non-null int64
险种代码_NIK11                  152420 non-null int64
就诊类型名称_一般门诊                 152420 non-null int64
就诊类型名称_中医药                  152420 non-null int64
就诊类型名称_其他约定                 152420 non-null int64
就诊类型名称_接种疫苗                 152420 non-null int64
就诊类型名称_普通生育门诊               152420 non-null int64
就诊类型名称_牙科治疗                 152420 non-null int64
就诊类型名称_精神疾病门诊               152420 non-null int64
就诊类型名称_统筹住院                 152420 non-null int64
就诊类型名称_统筹门诊                 152420 non-null int64
就诊类型名称_药房购药                 152420 non-null int64
就诊类型名称_门诊就诊                 152420 non-null int64
就诊类型名称_门诊意外就诊               152420 non-null int64
就诊类型名称_预防性检查                152420 non-null int64
费用金额                        152420 non-null float64
自费金额                        152420 non-null float64
部分自付金额                      152420 non-null float64
医保支付金额                      152420 non-null float64
费用项目名称_中成药费                 152420 non-null int64
费用项目名称_中草药                  152420 non-null int64
费用项目名称_其他费                  152420 non-null int64
费用项目名称_化验费                  152420 non-null int64
费用项目名称_床位费                  152420 non-null int64
费用项目名称_手术费                  152420 non-null int64
费用项目名称_护理费                  152420 non-null int64
费用项目名称_挂号费                  152420 non-null int64
费用项目名称_材料费                  152420 non-null int64
费用项目名称_检查费                  152420 non-null int64
费用项目名称_治疗费                  152420 non-null int64
费用项目名称_西药费                  152420 non-null int64
费用项目名称_诊疗费                  152420 non-null int64
event_num                   152420 non-null int64
hos_num                     152420 non-null int64
疾病代码_num                    152420 non-null int64
出险人客户号_num                  152420 non-null int64
自费总金额                       152420 non-null float64
自费总金额占比                     152420 non-null float64
医保支付金额占比                    152420 non-null float64
费用金额mean                    152420 non-null float64
费用金额log                     152420 non-null float64
自费金额log                     152420 non-null float64
部分自付金额log                   152420 non-null float64
医保支付金额log                   152420 non-null float64
自费总金额log                    152420 non-null float64
费用金额meanlog                 152420 non-null float64
dtypes: datetime64[ns](2), float64(15), int64(53), object(4)
memory usage: 87.2+ MB
        """
        mzdf.to_csv('../data/mz_all_claim.csv', encoding='gbk', index=True)

    if mode == 3:
        """
        收据压成事件
            同一时间、同一医院、同一科室（没有科室的话，就同一诊断）
            只要把费用和相对应的就诊类型名称相加即可
        """
        mzdf = pd.read_csv('../data/mz_all_receipt.csv', encoding='gbk', index_col=0, parse_dates=['出生日期', '就诊结帐费用发生日期'])
        flist = [
            '费用金额', '自费金额', '部分自付金额', '医保支付金额',
            '费用项目名称_中成药费', '费用项目名称_中草药', '费用项目名称_其他费', '费用项目名称_化验费',
            '费用项目名称_床位费', '费用项目名称_手术费', '费用项目名称_护理费', '费用项目名称_挂号费',
            '费用项目名称_材料费', '费用项目名称_检查费', '费用项目名称_治疗费', '费用项目名称_西药费',
            '费用项目名称_诊疗费'
        ]
        train_removedg = mzdf.groupby(['总案号_分案号', '医院代码', '疾病代码', '就诊结帐费用发生日期'])[flist].sum()
        mzdf = mzdf.drop_duplicates(subset=['总案号_分案号', '医院代码', '疾病代码', '就诊结帐费用发生日期'], keep='first')
        print(mzdf.shape)  # (267303, 60)
        for i in flist:
            del mzdf[i]
        mzdf = pd.merge(mzdf, train_removedg, how='left', left_on=['总案号_分案号', '医院代码', '疾病代码', '就诊结帐费用发生日期'], right_index=True)
        mzdf.to_csv('../data/mz_all_event.csv', encoding='gbk', index=True)
    if mode == 2:
        """
        明细压成收据
        只要把费用和相对应的就诊类型名称相加即可
        """
        mzdf = pd.read_csv('../data/mz_all.csv', index_col=0, encoding='gbk', parse_dates=['生效日期', '出生日期', '就诊结帐费用发生日期'])
        mzdf1 = pd.read_csv('../data/mz_all(1).csv', encoding='gbk', parse_dates=['生效日期', '出生日期', '就诊结帐费用发生日期'])
        print(mzdf1['就诊类型名称'].unique())
        mzdf['费用项目名称'] = mzdf1['费用项目名称']
        del mzdf1
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
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='牙科医疗'] = '牙科治疗'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='牙齿护理'] = '牙科治疗'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='牙科护理'] = '牙科治疗'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='牙科保健'] = '牙科治疗'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='紧急牙科治疗'] = '牙科治疗'

        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='普通生育门诊'] = '生育'

        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='门诊疾病就诊'] = '门诊就诊'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='统筹门诊'] = '门诊就诊'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='一般门诊'] = '门诊就诊'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='其他约定1门诊'] = '门诊就诊'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='门诊疾病就诊'] = '门诊就诊'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='门诊意外就诊'] = '门诊就诊'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='门诊意外首次就诊'] = '门诊就诊'

        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='住院'] = '住院就诊'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='统筹住院'] = '住院就诊'

        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='其他约定1'] = '其他约定'
        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='统筹约定'] = '其他约定'

        mzdf['就诊类型名称'][mzdf['就诊类型名称']=='中医药'] = '药房购药'
        print(mzdf['就诊类型名称'].value_counts())
        # mzdf['费用项目名称'][mzdf['费用项目名称']=='诊疗费'] = '治疗费'
        # mzdf['费用项目名称'][mzdf['费用项目名称']=='门诊手术费'] = '手术费'
        # mzdf['费用项目名称'][mzdf['费用项目名称']=='CT费'] = 'MRI/CT费'
        # mzdf['费用项目名称'][mzdf['费用项目名称']=='核磁费'] = 'MRI/CT费'
        # mzdf['费用项目名称'][mzdf['费用项目名称']=='放射费'] = 'MRI/CT费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='中成药'] = '中成药费'
        # mzdf['费用项目名称'][mzdf['费用项目名称']=='专家挂号费'] = '挂号费'
        # mzdf['费用项目名称'][mzdf['费用项目名称']=='合计金额'] = '其他费'
        # mzdf['费用项目名称'][mzdf['费用项目名称']=='氧气费'] = '输氧费'
        print(mzdf['费用项目名称'].value_counts())
        # print(mzdf['疾病代码'].value_counts())

        mzdf = de.build_one_hot_features(mzdf, ['保单号', '生效日期', '人员属性', '证件类型', '性别', '出险原因', '险种代码', '就诊类型名称', '费用项目名称'])
        mzdf.info()
        flist = [
            '费用金额', '自费金额', '部分自付金额', '医保支付金额',
            '费用项目名称_中成药费', '费用项目名称_中草药', '费用项目名称_其他费', '费用项目名称_化验费',
            '费用项目名称_床位费', '费用项目名称_手术费', '费用项目名称_护理费', '费用项目名称_挂号费',
            '费用项目名称_材料费', '费用项目名称_检查费', '费用项目名称_治疗费', '费用项目名称_西药费',
            '费用项目名称_诊疗费'
            ]
        train_removedg = mzdf.groupby(['总案号_分案号', '收据号'])[flist].sum()
        mzdf = mzdf.drop_duplicates(subset=['总案号_分案号', '收据号'], keep='first')
        print(mzdf.shape)#(657371, 60)
        for i in flist:
            del mzdf[i]
        mzdf = pd.merge(mzdf, train_removedg, how='left', left_on=['总案号_分案号', '收据号'], right_index=True)
        print(mzdf[mzdf['费用合计']-mzdf['费用金额'] > 0.001])
        mzdf.to_csv('../data/mz_all_receipt.csv', encoding='gbk', index=True)
    if mode == 1:
        """
        明细数据
        """
        mzdf4 = pd.read_csv('../data/mz4.csv').iloc[:, 2:]
        mzdf5 = pd.read_csv('../data/mz5.csv').iloc[:, 2:]
        mzdf6 = pd.read_csv('../data/mz6.csv').iloc[:, 2:]
        mzdf = pd.concat([mzdf4, mzdf5, mzdf6])
        print(mzdf['总案号'].unique().shape)#(16225,)
        print(mzdf['分案号'].unique().shape)#(152420,)
        mzdf['总案号_分案号'] = mzdf['总案号'].apply(lambda x: str(x) + '_') + mzdf['分案号'].apply(str)
        print(mzdf['总案号_分案号'].unique().shape)
        del mzdf['总案号']
        del mzdf['分案号']
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
        train_removed['医院等级'] = de.hosRank(train_removed['医院等级'])
        train_removed.info()
        train_removed.to_csv('../data/mz_all.csv', encoding='gbk', index=True)
        train_removed.iloc[:10000].to_csv('../data/mz_all1.csv', encoding='gbk', index=True)
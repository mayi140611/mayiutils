#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: data_explore.py
@time: 2019-05-06 17:22
"""
import pandas as pd
import math
import featuretools as ft
from feature_selector import FeatureSelector
from mayiutils.datasets.data_preprocessing import DataExplore as de


if __name__ == '__main__':
    mode = 2
    if mode == 7:
        """
        放弃利用featuretools扩充特征，自己构建特征
        """
        dfzy = pd.read_csv('zy_all_featured_claim.csv', parse_dates=['就诊结帐费用发生日期', '入院时间', '出院时间'],
                           encoding='gbk')

        del dfzy['ROWNUM']
        del dfzy['主被保险人客户号']
        del dfzy['出险人客户号']
        del dfzy['就诊结帐费用发生日期']
        del dfzy['自费描述']
        del dfzy['部分自付描述']
        del dfzy['医保支付描述']
        del dfzy['出院时间']
        del dfzy['event_id']

        # 构造特征
        # 自费金额占比
        dfzy['自费总金额'] = dfzy['自费金额'] + dfzy['部分自付金额']
        # 自费总金额占费用金额比
        dfzy['自费总金额占比'] = dfzy['自费总金额'] / dfzy['费用金额']
        # 医保支付金额占比
        dfzy['医保支付金额占比'] = dfzy['医保支付金额'] / dfzy['费用金额']
        # 平均每次事件费用金额
        dfzy['费用金额mean'] = dfzy['费用金额'] / dfzy['event_count']
        # log
        def tlog(x):
            if x < 1:
                x = 0
            if x != 0:
                x = math.log(x)
            return x
        dfzy['费用金额log'] = dfzy['费用金额'].apply(tlog)
        dfzy['自费金额log'] = dfzy['自费金额'].apply(tlog)
        dfzy['部分自付金额log'] = dfzy['部分自付金额'].apply(tlog)
        dfzy['医保支付金额log'] = dfzy['医保支付金额'].apply(tlog)
        dfzy['自费总金额log'] = dfzy['自费总金额'].apply(tlog)
        dfzy['费用金额meanlog'] = dfzy['费用金额mean'].apply(tlog)
        # 构建one-hot特征

        def build_one_hot_features(df, cols):
            for col in cols:
                t = pd.get_dummies(dfzy[col], prefix=col)
                df = pd.concat([df, t], axis=1)
                del df[col]
            return df
        del dfzy['疾病代码']
        del dfzy['医院代码']
        # dfzy['疾病代码'] = dfzy['疾病代码'].apply(lambda x: x[:3])
        # s = dfzy['疾病代码'].value_counts()
        # dfzy['疾病代码'][dfzy['疾病代码'].isin(list(s[s < 40].index))] = 'other'
        # print(dfzy['医院代码'].value_counts())
        def t(x):
            if x.find('三级') != -1:
                return 3
            if x.find('二级') != -1:
                return 2
            if x.find('一级') != -1:
                return 1
            if x.find('未评级') != -1:
                return 0
        # print(dfzy['医院等级'].value_counts())
        dfzy['医院等级'] = dfzy['医院等级'].apply(t)
        # dfzy = build_one_hot_features(dfzy, ['性别', '证件类型', '人员属性', '出险原因', '险种代码', '医院代码', '费用项目代码', '就诊类型名称', '医院等级', '疾病代码'])
        dfzy = build_one_hot_features(dfzy, ['性别', '证件类型', '人员属性', '出险原因', '险种代码', '费用项目代码', '就诊类型名称'])

        # print(s[s<40].index)
        # print(dfzy['疾病代码'].value_counts())
        # dfzy.info()
        fs = FeatureSelector(data=dfzy)
        fs.identify_collinear(correlation_threshold=0.975)
        """
        2 features with a correlation magnitude greater than 0.97.
        """
        correlated_features = fs.ops['collinear']
        # print(correlated_features)
        print(fs.record_collinear.head(30))
        train_removed_all_once = fs.remove(methods=['collinear'])
        train_removed_all_once.index = train_removed_all_once['总案号_分案号']
        del train_removed_all_once['总案号_分案号']
        del train_removed_all_once['入院时间']
        print(train_removed_all_once.shape)#(9843, 350)
        print(list(train_removed_all_once.columns))
        train_removed_all_once.info()
        # print(train_removed_all_once.index)
        train_removed_all_once.to_csv('zy_train_data.csv', encoding='gbk')
    if mode == 6:
        """
        featuretools扩充特征 基本没用啊
        """
        dfzy = pd.read_csv('zy_all_featured_claim.csv', parse_dates=['出生日期', '就诊结帐费用发生日期', '入院时间', '出院时间'], encoding='gbk')

        del dfzy['ROWNUM']
        del dfzy['主被保险人客户号']
        del dfzy['出险人客户号']
        del dfzy['就诊结帐费用发生日期']
        del dfzy['自费描述']
        del dfzy['部分自付描述']
        del dfzy['医保支付描述']
        del dfzy['出院时间']
        del dfzy['event_id']
        # dfzy.info()
        es = ft.EntitySet(id='zy')
        es = es.entity_from_dataframe(entity_id='zy',
                                      dataframe=dfzy,
                                      variable_types={
                                          '人员属性': ft.variable_types.Categorical,
                                          '证件类型': ft.variable_types.Categorical,
                                          '费用项目代码': ft.variable_types.Categorical,
                                          '生效年': ft.variable_types.Categorical,
                                          },
                                      index='总案号_分案号',
                                      time_index='出生日期')
        # print(es)
        # print(es['zy'])
        # Perform deep feature synthesis without specifying primitives
        features, feature_names = ft.dfs(entityset=es, target_entity='zy',
                                         max_depth=2)
        # print(feature_names)
        # print(type(features))
        print(features.shape)
        # print(features.head())
        # features.to_csv('zy_all_featured_claim_derivation.csv', encoding='gbk', index=False)
        fs = FeatureSelector(data=features)
        fs.identify_collinear(correlation_threshold=0.975)
        """
        2 features with a correlation magnitude greater than 0.97.
        """
        correlated_features = fs.ops['collinear']
        print(correlated_features[:5])
        print(fs.record_collinear.head())
        train_removed_all_once = fs.remove(methods=['collinear'])
        print(type(fs.data))
        print(type(train_removed_all_once))
        print(train_removed_all_once.shape)
        train_removed_all_once.info()
    if mode == 5:
        """
        事件压成赔案
        """
        dfzy = pd.read_csv('zy_all_featured_event.csv', parse_dates=['就诊结帐费用发生日期', '入院时间', '出院时间'], encoding='gbk')
        del dfzy['收据号']
        dfzyg = dfzy.groupby(['总案号_分案号'])[['费用金额', '自费金额', '部分自付金额', '医保支付金额']].sum()
        print(dfzyg.shape)  # (9843, 4)
        dfzyg1 = dfzy.groupby(['总案号_分案号'])[['event_id']].count()
        dfzyg1.columns = ['event_count']
        print(dfzyg1.shape)  # (9843, 4)
        print(dfzyg1.head())  # (9843, 4)
        dfzy1 = dfzy.drop_duplicates(subset=['总案号_分案号'], keep='first')
        print(dfzy1.shape)
        del dfzy1['费用金额']
        del dfzy1['自费金额']
        del dfzy1['部分自付金额']
        del dfzy1['医保支付金额']
        dfzy2 = pd.merge(dfzy1, dfzyg, how='left', left_on=['总案号_分案号'], right_index=True)
        dfzy2 = pd.merge(dfzy2, dfzyg1, how='left', left_on=['总案号_分案号'], right_index=True)
        dfzy2.to_csv('zy_all_featured_claim.csv', encoding='gbk', index=False)

    if mode == 4:
        """
        收据压成事件
            同一时间、同一医院、同一科室（没有科室的话，就同一诊断）
        """
        dfzy = pd.read_csv('zy_all_featured_receipt.csv', parse_dates=['就诊结帐费用发生日期', '入院时间', '出院时间'], encoding='gbk')
        dfzyg = dfzy.groupby(['总案号_分案号', '医院代码', '疾病代码', '入院时间'])[['费用金额', '自费金额', '部分自付金额', '医保支付金额']].sum()
        print(dfzyg.shape)#(10278, 4)
        dfzy1 = dfzy.drop_duplicates(subset=['总案号_分案号', '医院代码', '疾病代码', '入院时间'], keep='first')
        print(dfzy1.shape)
        del dfzy1['费用金额']
        del dfzy1['自费金额']
        del dfzy1['部分自付金额']
        del dfzy1['医保支付金额']
        dfzy2 = pd.merge(dfzy1, dfzyg, how='left', left_on=['总案号_分案号', '医院代码', '疾病代码', '入院时间'], right_index=True)
        dfzy2['event_id'] = list(range(dfzy2.shape[0]))
        dfzy2.to_csv('zy_all_featured_event.csv', encoding='gbk', index=False)
    if mode == 3:
        
        """
        收据压成事件
            同一时间、同一医院、同一科室（没有科室的话，就同一诊断）
            只要把费用和相对应的就诊类型名称相加即可
        """
        zydf = pd.read_csv('../data/zy_all_receipt2.csv', encoding='gbk', index_col=0, parse_dates=['出生日期', '入院时间', '出院时间'])
        flist = [
            '费用金额', '自费金额', '部分自付金额', '医保支付金额',
            '中成药费', '中草药', '其他费', '化验费',
            '床位费', '手术费', '护理费', '挂号费',
            '材料费', '检查费', '治疗费', '西药费',
            '诊疗费'
        ]
        train_removedg = zydf.groupby(['总案号_分案号', '医院代码', '疾病代码', '入院时间'])[flist].sum()
        zydf = zydf.drop_duplicates(subset=['总案号_分案号', '医院代码', '疾病代码', '入院时间'], keep='first')
        print(zydf.shape)  # (267303, 43)
        for i in flist:
            del zydf[i]
        zydf = pd.merge(zydf, train_removedg, how='left', left_on=['总案号_分案号', '医院代码', '疾病代码', '入院时间'], right_index=True)

        del zydf['费用合计']
        flist = [
            '中成药费', '中草药', '其他费', '化验费',
            '床位费', '手术费', '护理费', '挂号费',
            '材料费', '检查费', '治疗费', '西药费',
            '诊疗费'
        ]

        # 构造特征
        # 自费金额占比
        # 自费总金额占费用金额比
        zydf['自费总金额占比'] = (zydf['自费金额'] + zydf['部分自付金额']) / zydf['费用金额']
        # 医保支付金额占比
        zydf['医保支付金额占比'] = zydf['医保支付金额'] / zydf['费用金额']
        # 费用项目占比
        zydf['药费占比'] = (zydf['中成药费'] + zydf['中草药'] + zydf['西药费']) / zydf['费用金额']
        # 检查费占比
        zydf['检查费占比'] = (zydf['检查费']) / zydf['费用金额']
        zydf['床位费占比'] = (zydf['床位费']) / zydf['费用金额']
        zydf['手术费占比'] = (zydf['手术费']) / zydf['费用金额']
        zydf['护理费占比'] = (zydf['护理费']) / zydf['费用金额']
        zydf['治疗费占比'] = (zydf['治疗费']) / zydf['费用金额']
        zydf['诊疗费占比'] = (zydf['诊疗费']) / zydf['费用金额']
        zydf['化验费占比'] = (zydf['化验费']) / zydf['费用金额']
        for i in flist:
            del zydf[i]

        print(zydf.shape)  # (267303, 39)
        zydf.info()
        """

        """
        zydf.to_csv('../data/zy_all_event2.csv', encoding='gbk', index=True)
    if mode == 2:
        """
        明细压成收据
        """
        zydf = pd.read_csv('../data/zy_all2.csv', index_col=0, encoding='gbk',
                           parse_dates=['生效日期', '出生日期', '就诊结帐费用发生日期', '入院时间', '出院时间'])
        del zydf['就诊结帐费用发生日期']
        zydf = zydf[zydf['入院时间'].notnull()]
        # 构造特征：住院天数
        zydf['住院天数'] = [t.days + 1 for t in (zydf['出院时间'] - zydf['入院时间'])]
        zydf['费用项目名称'][zydf['费用项目名称'] == '中成药'] = '中成药费'
        zydf['费用项目名称'][zydf['费用项目名称'] == '中成药'] = '中成药费'
        zydf['费用项目名称'][zydf['费用项目名称'] == '中成药'] = '中成药费'
        zydf['费用项目名称'][zydf['费用项目名称'] == '中成药'] = '中成药费'
        zydf['费用项目名称'][zydf['费用项目名称'] == '中成药'] = '中成药费'
        zydf['费用项目名称'][zydf['费用项目名称'] == '中成药'] = '中成药费'
        print(zydf['费用项目名称'].value_counts())

        flist = [
            '费用金额', '自费金额', '部分自付金额', '医保支付金额'
        ]
        train_removedg = zydf.groupby(['总案号_分案号', '收据号'])[flist].sum()
        zydf['总案号_分案号_收据号'] = zydf['总案号_分案号'].apply(lambda x: x + '_') + zydf['收据号']
        df = zydf.groupby(['总案号_分案号_收据号', '费用项目名称'])['费用金额'].sum()
        del zydf['费用金额']
        zydf = zydf.drop_duplicates(subset=['总案号_分案号_收据号', '费用项目名称'], keep='first')
        zydf = pd.merge(zydf, df, left_on=['总案号_分案号_收据号', '费用项目名称'], right_index=True)
        df = zydf.pivot(index='总案号_分案号_收据号', columns='费用项目名称', values='费用金额').fillna(0)
        del zydf['费用项目名称']
        zydf = zydf.drop_duplicates(subset=['总案号_分案号', '收据号'], keep='first')
        zydf = pd.merge(zydf, df, how='left', left_on='总案号_分案号_收据号', right_index=True)
        del zydf['总案号_分案号_收据号']
        del df
        print(zydf.shape)  # (10394, 52)
        for i in flist:
            del zydf[i]
        zydf = pd.merge(zydf, train_removedg, how='left', left_on=['总案号_分案号', '收据号'], right_index=True)
        print(zydf[zydf['费用合计'] - zydf['费用金额'] > 0.001])
        zydf.info()
        """
        """
        zydf.to_csv('../data/zy_all_receipt2.csv', encoding='gbk', index=True)
    if mode == 1:
        zydf = pd.read_csv('../data/zy_all.csv').iloc[:, 3:]
        print(zydf['总案号'].unique().shape)  # (2699,)
        print(zydf['分案号'].unique().shape)  # (9844,)
        zydf['总案号_分案号'] = zydf['总案号'].apply(lambda x: str(x) + '_') + zydf['分案号'].apply(str)
        print(zydf['总案号_分案号'].unique().shape)  # (9844,)
        del zydf['总案号']
        del zydf['分案号']
        fs = FeatureSelector(data=zydf)
        fs.identify_missing(missing_threshold=0.2)
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
5                            自费描述          0.504085
6                          部分自付描述          0.443064
7                          医保支付描述          0.362454
        """
        fs.identify_single_unique()
        print(fs.record_single_unique)
        """
        """
        fs.identify_collinear(correlation_threshold=0.975)
        print(fs.record_collinear)
        """
        """
        train_removed = fs.remove(methods=['missing', 'single_unique', 'collinear'])
        train_removed['医院等级'] = de.hosRank(train_removed['医院等级'])
        train_removed.info()
        """
        """
        train_removed.to_csv('../data/zy_all2.csv', encoding='gbk', index=True)
    if mode == 0:
        """
        可以看到，zy数据比m z数据少很多，先拿住院数据进行分析
        """
        dfzy4 = pd.read_csv('zy4.csv')
        dfzy5 = pd.read_csv('zy5.csv')
        dfzy6 = pd.read_csv('zy6.csv')
        dfzy = pd.concat([dfzy4, dfzy5, dfzy6])
        print(dfzy.shape)#(11996, 46)
        dfzy.to_csv('zy_all.csv')

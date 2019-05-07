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


if __name__ == '__main__':
    mode = 7
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
        明细压成收据
        """
        dfzy = pd.read_csv('zy_all_featured.csv', parse_dates=['就诊结帐费用发生日期', '入院时间', '出院时间'], encoding='gbk')
        dfzyg = dfzy.groupby(['总案号_分案号', '收据号'])[['费用金额', '自费金额', '部分自付金额', '医保支付金额']].sum()
        print(dfzyg.head())
        print(dfzyg.shape)#(10393, 4)
        dfzy1 = dfzy.drop_duplicates(subset=['总案号_分案号', '收据号'], keep='first')
        print(dfzy1.shape)
        del dfzy1['费用金额']
        del dfzy1['自费金额']
        del dfzy1['部分自付金额']
        del dfzy1['医保支付金额']
        dfzy2 = pd.merge(dfzy1, dfzyg, how='left', left_on=['总案号_分案号', '收据号'], right_index=True)
        print(dfzy2.head())
        print(dfzy2.shape)#(10393, 30)
        print(dfzy2['总案号_分案号'].unique().shape)#(9843,)
        print(dfzy2[dfzy2['费用合计'] - dfzy2['费用金额'] > 0.01].shape)#(0, 30)
        del dfzy2['费用合计']

        dfzy2.to_csv('zy_all_featured_receipt.csv', encoding='gbk', index=False)
    if mode == 2:
        """
        住院数据特征explore
        """
        dfzy = pd.read_csv('zy_all.csv', parse_dates=['生效日期', '出生日期', '就诊结帐费用发生日期', '入院时间', '出院时间']).dropna(axis=1, how='all').iloc[:, 3:]
        # dfzy.info()
        """
RangeIndex: 11996 entries, 0 to 11995
Data columns (total 39 columns):
ROWNUM        11996 non-null int64
保单号           11996 non-null int64
投保单位名称        11996 non-null object
生效日期          11996 non-null datetime64[ns]
总案号           11996 non-null int64
分案号           11996 non-null int64
出险人姓名         11996 non-null object
主被保险人         11996 non-null object
主被保险人客户号      11996 non-null int64
人员属性          11996 non-null object
证件类型          11996 non-null int64
出险人客户号        11996 non-null int64
性别            11996 non-null object
年龄            11996 non-null int64
出生日期          11996 non-null datetime64[ns]
出险原因          11996 non-null object
险种名称          11996 non-null object
险种代码          11996 non-null object
收据号           11996 non-null object
医院代码          11996 non-null object
医院名称          11996 non-null object
医院等级          11996 non-null object
就诊结帐费用发生日期    10164 non-null datetime64[ns]
费用合计          11996 non-null float64
费用项目代码        11996 non-null int64
费用项目名称        11996 non-null object
费用金额          11996 non-null float64
疾病代码          11996 non-null object
疾病名称          11996 non-null object
就诊类型          11996 non-null object
就诊类型名称        11893 non-null object
自费描述          5949 non-null object
自费金额          11996 non-null float64
部分自付描述        6681 non-null object
部分自付金额        11996 non-null float64
医保支付金额        11996 non-null float64
医保支付描述        7648 non-null object
入院时间          11995 non-null datetime64[ns]
出院时间          11995 non-null datetime64[ns]
dtypes: datetime64[ns](5), float64(5), int64(9), object(20)
memory usage: 3.6+ MB
        """
        dfzy = dfzy[dfzy['入院时间'].notnull()]
        # 总案号_分案号
        dfzy['总案号_分案号'] = dfzy['总案号'].apply(lambda x: str(x)+'_') + dfzy['分案号'].apply(str)
        print(dfzy['总案号_分案号'].unique().shape)#(9843,)
        del dfzy['总案号']
        del dfzy['分案号']
        # 和客户号重复
        del dfzy['出险人姓名']
        del dfzy['主被保险人']

        # 和 险种代码重复
        del dfzy['险种名称']
        # 和医院代码重复
        del dfzy['医院名称']
        del dfzy['费用项目名称']
        del dfzy['疾病名称']

        # 唯一值
        del dfzy['就诊类型']
        del dfzy['投保单位名称']

        dfzy['生效年'] = dfzy['生效日期'].dt.year
        del dfzy['生效日期']
        del dfzy['保单号']

        dfzy['出生月'] = dfzy['出生日期'].dt.month
        del dfzy['出生日期']

        dfzy['住院weekday'] = dfzy['入院时间'].dt.weekday
        # 构造特征：住院天数
        dfzy['住院天数'] = [t.days+1 for t in (dfzy['出院时间'] - dfzy['入院时间'])]
        # print(dfzy['医保支付描述'].value_counts())
        # print(dfzy['部分自付描述'].value_counts())
        # print(dfzy['自费描述'].value_counts())
        print(dfzy['性别'].value_counts())
        print(dfzy['收据号'].value_counts())
        # print(dfzy['住院天数'])
        # dfzy.info()
        dfzy.to_csv('zy_all_featured.csv', encoding='gbk', index=False)
    if mode == 1:
        """
        可以看到，zy数据比mz数据少很多，先拿住院数据进行分析
        """
        dfzy4 = pd.read_csv('zy4.csv')
        dfzy5 = pd.read_csv('zy5.csv')
        dfzy6 = pd.read_csv('zy6.csv')
        dfzy = pd.concat([dfzy4, dfzy5, dfzy6])
        print(dfzy.shape)#(11996, 46)
        dfzy.to_csv('zy_all.csv')
        # df6 = pd.read_excel('/Users/luoyonggui/Documents/datasets/work/3/82200946506.xlsx')
        # dfmz = df6[df6['就诊类型']=='门诊']
        # print(dfmz.shape)#(301577, 45)
        # dfmz.to_csv('mz6.csv')
        # dfzy = df6[df6['就诊类型']=='住院']
        # print(dfzy.shape)#(3075, 45)
        # dfzy.to_csv('zy6.csv')
        # df5 = pd.read_excel('/Users/luoyonggui/Documents/datasets/work/3/82200946505.xlsx')
        # dfmz = df5[df5['就诊类型']=='门诊']
        # print(dfmz.shape)#(343772, 45)
        # dfmz.to_csv('mz5.csv')
        # dfzy = df5[df5['就诊类型']=='住院']
        # print(dfzy.shape)#(4814, 45)
        # dfzy.to_csv('zy5.csv')
        # df4 = pd.read_excel('/Users/luoyonggui/Documents/datasets/work/3/82200946504.xlsx')
        # dfmz = df4[df4['就诊类型']=='门诊']
        # print(dfmz.shape)#(350922, 45)
        # dfmz.to_csv('mz4.csv')
        # dfzy = df4[df4['就诊类型']=='住院']
        # print(dfzy.shape)#(4107, 45)
        # dfzy.to_csv('zy4.csv')
        # df4.info()
        """
RangeIndex: 355029 entries, 0 to 355028
Data columns (total 45 columns):
Unnamed: 0                        355029 non-null int64
ROWNUM                            355029 non-null int64
保单号                               355029 non-null int64
投保单位名称                            355029 non-null object
生效日期                              355029 non-null datetime64[ns]
(SELECTDISTINCT(P.PAYENDDATE-1    0 non-null float64
总案号                               355029 non-null int64
分案号                               355029 non-null int64
出险人姓名                             355029 non-null object
主被保险人                             355029 non-null object
主被保险人客户号                          355029 non-null int64
人员属性                              355029 non-null object
证件类型                              355029 non-null int64
身份标识                              0 non-null float64
身份备注                              0 non-null float64
证件号码                              0 non-null float64
出险人客户号                            355029 non-null int64
性别                                355029 non-null object
年龄                                355029 non-null int64
出生日期                              355029 non-null datetime64[ns]
出险原因                              355029 non-null object
险种名称                              355029 non-null object
险种代码                              355029 non-null object
发票类型                              0 non-null float64
收据号                               355029 non-null object
医院代码                              355029 non-null object
医院名称                              355029 non-null object
医院等级                              355029 non-null object
就诊结帐费用发生日期                        354058 non-null datetime64[ns]
费用合计                              355029 non-null float64
费用项目代码                            355029 non-null int64
费用项目名称                            355029 non-null object
费用金额                              355029 non-null float64
疾病代码                              355029 non-null object
疾病名称                              355029 non-null object
就诊类型                              355029 non-null object
就诊类型名称                            354625 non-null object
自费描述                              100835 non-null object
自费金额                              355029 non-null float64
部分自付描述                            28997 non-null object
部分自付金额                            355029 non-null float64
医保支付金额                            355029 non-null float64
医保支付描述                            7265 non-null object
入院时间                              4107 non-null datetime64[ns]
出院时间                              4107 non-null datetime64[ns]
dtypes: datetime64[ns](5), float64(10), int64(10), object(20)
memory usage: 121.9+ MB     
        """
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
    """
    第二次迭代
    """
    mode = 8
    if mode == 9:
        """
        三个点需要关注：
            1、哪些人在牙科医疗就诊比较多；
            2、哪些人慢病门诊就诊如糖尿病，但是住院就诊诊断与门诊不一致的；
            3、哪些人工作日就诊比较多。
        """
        mzdf = pd.read_csv('../data/mz_all_event2.csv', encoding='gbk', index_col=0, parse_dates=['就诊结帐费用发生日期'])
        mzdf['weekday'] = mzdf['就诊结帐费用发生日期'].dt.weekday + 1
        print(mzdf[['就诊结帐费用发生日期', 'weekday']])
        dic = dict()
        for line in mzdf[['出险人客户号', '出险人姓名']].itertuples():
            if line[1] not in dic:
                dic[line[1]] = line[2]

        mzdf.loc[mzdf['就诊类型名称'].isin(['牙科医疗', '牙齿护理', '牙科护理', '牙科保健', '紧急牙科治疗']), '就诊类型名称'] = '牙科治疗'
        mzdf1 = mzdf[mzdf['就诊类型名称']=='牙科治疗']
        df = mzdf1
        dfg = df.groupby(['生效日期', '出险人客户号'])
        dff = pd.DataFrame()

        dft = dfg['ROWNUM'].count()
        dft.name = 'count'
        dff['2016就诊次数牙科前十'] = list(dft.loc['2016-06-01'].sort_values(ascending=False)[:10].index)
        dff['2016就诊次数牙科前十姓名'] = dff['2016就诊次数牙科前十'].apply(lambda x: dic[x])
        dff['2016就诊次数牙科'] = list(dft.loc['2016-06-01'].sort_values(ascending=False)[:10])
        dff['2017就诊次数牙科前十'] = list(dft.loc['2017-06-01'].sort_values(ascending=False)[:10].index)
        dff['2017就诊次数牙科前十姓名'] = dff['2017就诊次数牙科前十'].apply(lambda x: dic[x])
        dff['2017就诊次数牙科'] = list(dft.loc['2017-06-01'].sort_values(ascending=False)[:10])
        dff['2018就诊次数牙科前十'] = list(dft.loc['2018-06-01'].sort_values(ascending=False)[:10].index)
        dff['2018就诊次数牙科前十姓名'] = list(dft.loc['2018-06-01'].sort_values(ascending=False)[:10].index)
        dff['2018就诊次数牙科'] = dff['2018就诊次数牙科前十'].apply(lambda x: dic[x])

        mzdf1 = mzdf[mzdf['weekday'].isin([6, 7])==False]
        df = mzdf1
        dfg = df.groupby(['生效日期', '出险人客户号'])
        dft = dfg['ROWNUM'].count()
        dft.name = 'count'
        dff['2016就诊次数工作日前十'] = list(dft.loc['2016-06-01'].sort_values(ascending=False)[:10].index)
        dff['2016就诊次数工作日前十姓名'] = dff['2016就诊次数工作日前十'].apply(lambda x: dic[x])
        dff['2016就诊次数工作日'] = list(dft.loc['2016-06-01'].sort_values(ascending=False)[:10])
        dff['2017就诊次数工作日前十'] = list(dft.loc['2017-06-01'].sort_values(ascending=False)[:10].index)
        dff['2017就诊次数工作日前十姓名'] = dff['2017就诊次数工作日前十'].apply(lambda x: dic[x])
        dff['2017就诊次数工作日'] = list(dft.loc['2017-06-01'].sort_values(ascending=False)[:10])
        dff['2018就诊次数工作日前十'] = list(dft.loc['2018-06-01'].sort_values(ascending=False)[:10].index)
        dff['2018就诊次数工作日前十姓名'] = list(dft.loc['2018-06-01'].sort_values(ascending=False)[:10].index)
        dff['2018就诊次数工作日'] = dff['2018就诊次数工作日前十'].apply(lambda x: dic[x])

        dff.to_excel('r1.xlsx')
    if mode == 8:
        """
        按年度统计 就诊次数前十和账单金额前十         
        """
        df = pd.read_csv('../data/mz_all_event2.csv', encoding='gbk', index_col=0)
        df.info()
        dic = dict()
        for line in df[['出险人客户号', '出险人姓名']].itertuples():
            if line[1] not in dic:
                dic[line[1]] = line[2]
        dfg = df.groupby(['生效日期', '出险人客户号'])
        df1 = df.drop_duplicates(['生效日期', '出险人客户号'])
        dff = pd.DataFrame()

        dft = dfg['ROWNUM'].count()
        dft.name = 'count'
        # print(dft.loc['2016-06-01'].sort_values(ascending=False)[:10])
        # print(dft.loc['2017-06-01'].sort_values(ascending=False)[:10])
        # print(dft.loc['2018-06-01'].sort_values(ascending=False)[:10])

        dff['2016就诊次数前十'] = list(dft.loc['2016-06-01'].sort_values(ascending=False)[:10].index)
        dff['2016就诊次数前十姓名'] = dff['2016就诊次数前十'].apply(lambda x: dic[x])
        dff['2016就诊次数'] = list(dft.loc['2016-06-01'].sort_values(ascending=False)[:10])
        dff['2017就诊次数前十'] = list(dft.loc['2017-06-01'].sort_values(ascending=False)[:10].index)
        dff['2017就诊次数前十姓名'] = dff['2017就诊次数前十'].apply(lambda x: dic[x])
        dff['2017就诊次数'] = list(dft.loc['2017-06-01'].sort_values(ascending=False)[:10])
        dff['2018就诊次数前十'] = list(dft.loc['2018-06-01'].sort_values(ascending=False)[:10].index)
        dff['2018就诊次数前十姓名'] = list(dft.loc['2018-06-01'].sort_values(ascending=False)[:10].index)
        dff['2018就诊次数'] = dff['2018就诊次数前十'].apply(lambda x: dic[x])
        dft = dfg['费用金额'].sum()

        # print(dft.loc['2016-06-01'].sort_values(ascending=False)[:10])
        # print(dft.loc['2017-06-01'].sort_values(ascending=False)[:10])
        # print(dft.loc['2018-06-01'].sort_values(ascending=False)[:10])

        dff['2016费用金额前十'] = list(dft.loc['2016-06-01'].sort_values(ascending=False)[:10].index)
        dff['2016费用金额前十姓名'] = dff['2016费用金额前十'].apply(lambda x: dic[x])
        dff['2016费用金额'] = list(dft.loc['2016-06-01'].sort_values(ascending=False)[:10])
        dff['2017费用金额前十'] = list(dft.loc['2017-06-01'].sort_values(ascending=False)[:10].index)
        dff['2017费用金额前十姓名'] = dff['2017费用金额前十'].apply(lambda x: dic[x])
        dff['2017费用金额'] = list(dft.loc['2017-06-01'].sort_values(ascending=False)[:10])
        dff['2018费用金额前十'] = list(dft.loc['2018-06-01'].sort_values(ascending=False)[:10].index)
        dff['2018费用金额前十姓名'] = dff['2018费用金额前十'].apply(lambda x: dic[x])
        dff['2018费用金额'] = list(dft.loc['2018-06-01'].sort_values(ascending=False)[:10])

        # print(dff.head())

        dff.to_excel('r.xlsx')
        # df1 = pd.merge(df1, dft, left_on=['生效日期', '出险人客户号'], right_index=True)
        # print(dfg['ROWNUM'].count())
        # print(df.loc['2016-06-01'])
    if mode == 7:
        """
        
        """
        mzdf1 = pd.read_csv('../data/mz_risk_taker_pred_iforest2.csv', encoding='gbk', index_col=0)
        # mzdf1.info()
        print(list(mzdf1.columns))
        # print(mzdf1.index)
        # mzdf1['event_num'].sort_values(ascending=False)[:10].to_excel('就诊次数前十.xlsx')
        # mzdf1["('event_num_per_A', Period('2016', 'A-DEC'))"].sort_values(ascending=False)[:10].to_excel('就诊次数前十2016.xlsx')
        # mzdf1["('event_num_per_A', Period('2017', 'A-DEC'))"].sort_values(ascending=False)[:10].to_excel('就诊次数前十2017.xlsx')
        # mzdf1["('event_num_per_A', Period('2018', 'A-DEC'))"].sort_values(ascending=False)[:10].to_excel('就诊次数前十2018.xlsx')
        # mzdf1["('event_num_per_A', Period('2019', 'A-DEC'))"].sort_values(ascending=False)[:10].to_excel('就诊次数前十2019.xlsx')


        mzdf1['费用金额'].sort_values(ascending=False)[:10].to_excel('费用金额前十.xlsx')
        mzdf1["('费用金额_per_A', Period('2016', 'A-DEC'))"].sort_values(ascending=False)[:10].to_excel('费用金额前十2016.xlsx')
        mzdf1["('费用金额_per_A', Period('2017', 'A-DEC'))"].sort_values(ascending=False)[:10].to_excel('费用金额前十2017.xlsx')
        mzdf1["('费用金额_per_A', Period('2018', 'A-DEC'))"].sort_values(ascending=False)[:10].to_excel('费用金额前十2018.xlsx')
        mzdf1["('费用金额_per_A', Period('2019', 'A-DEC'))"].sort_values(ascending=False)[:10].to_excel('费用金额前十2019.xlsx')
    if mode == 6:
        # mzdf1 = pd.read_csv('../data/mz_rm_tooth_pred_iforest2.csv', encoding='gbk', index_col=0)
        # del mzdf1['收据号']
        # mzdf1.head(10).to_excel('../data/mz_rm_tooth_pred_iforest2_pre10.xlsx', index=True)
        mzdf1 = pd.read_csv('../data/mz_risk_taker_pred_iforest2.csv', encoding='gbk', index_col=0)
        mzdf1.head(10).to_excel('../data/mz_risk_taker_pred_iforest2_pre10.xlsx', index=True)
    if mode == 4:
        """
        发现从赔案的层面的意义不大
        事件压成赔案
            把费用和相对应的就诊类型名称相加
            统计每个赔案包含的事件数
            统计每个赔案的去医院数量
            统计每个赔案的诊断数量
            统计每个赔案的出险人数量
            统计每个赔案去的不同等级的医院数量 暂时没做！
            各种费用占总费用比
            各种费用的log（去掉原费用）
            每个事件的mean、max、min
        """
        mzdf = pd.read_csv('../data/mz_all_event2.csv', encoding='gbk', index_col=0,
                           parse_dates=['出生日期', '就诊结帐费用发生日期'])
        flist = [
            '费用金额', '自费金额', '部分自付金额', '医保支付金额',
            '中成药费', '中草药', '其他费', '化验费',
            '床位费', '手术费', '护理费', '挂号费',
            '材料费', '检查费', '治疗费', '西药费',
            '诊疗费'
        ]
        train_removedg = mzdf.groupby(['总案号_分案号'])[flist].sum()
        train_removedg1 = mzdf.groupby(['总案号_分案号'])['收据号'].count()
        train_removedg1.columns = ['event_num']
        train_removedg1.name = 'event_num'

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
        print(type(train_removedg3))
        print(train_removedg3.value_counts())
        # 统计出险人的数量
        train_removedg4 = mzdf.groupby(['总案号_分案号'])['出险人客户号'].agg(t)
        train_removedg4.columns = ['出险人客户号_num']
        train_removedg4.name = '出险人客户号_num'
        print(train_removedg4.value_counts())
        # 事件最大费用金额
        train_removedg5 = mzdf.groupby(['总案号_分案号'])['费用金额'].max()
        train_removedg5.columns = ['event费用金额_max']
        train_removedg5.name = 'event费用金额_max'

        mzdf = mzdf.drop_duplicates(subset=['总案号_分案号'], keep='first')
        for i in flist:
            del mzdf[i]
        mzdf = pd.merge(mzdf, train_removedg, how='left', left_on=['总案号_分案号'], right_index=True)
        mzdf = pd.merge(mzdf, train_removedg1, how='left', left_on=['总案号_分案号'], right_index=True)
        mzdf = pd.merge(mzdf, train_removedg2, how='left', left_on=['总案号_分案号'], right_index=True)
        mzdf = pd.merge(mzdf, train_removedg3, how='left', left_on=['总案号_分案号'], right_index=True)
        mzdf = pd.merge(mzdf, train_removedg4, how='left', left_on=['总案号_分案号'], right_index=True)
        mzdf = pd.merge(mzdf, train_removedg5, how='left', left_on=['总案号_分案号'], right_index=True)
        del train_removedg, train_removedg1, train_removedg2, train_removedg3, train_removedg4, train_removedg5
        # 构造特征
        # 自费金额占比
        mzdf['自费总金额'] = mzdf['自费金额'] + mzdf['部分自付金额']
        # 自费总金额占费用金额比
        mzdf['自费总金额占比'] = mzdf['自费总金额'] / mzdf['费用金额']
        # 医保支付金额占比
        mzdf['医保支付金额占比'] = mzdf['医保支付金额'] / mzdf['费用金额']
        # 平均每次事件费用金额
        mzdf['event费用金额mean'] = mzdf['费用金额'] / mzdf['event_num']
        # 费用项目占比
        mzdf['药费占比'] = (mzdf['中成药费'] + mzdf['中草药'] + mzdf['西药费']) / mzdf['费用金额']
        # 检查费占比
        mzdf['检查费占比'] = (mzdf['检查费']) / mzdf['费用金额']
        mzdf['床位费占比'] = (mzdf['床位费']) / mzdf['费用金额']
        mzdf['手术费占比'] = (mzdf['手术费']) / mzdf['费用金额']
        mzdf['护理费占比'] = (mzdf['护理费']) / mzdf['费用金额']
        mzdf['治疗费占比'] = (mzdf['治疗费']) / mzdf['费用金额']
        mzdf['诊疗费占比'] = (mzdf['诊疗费']) / mzdf['费用金额']
        mzdf['化验费占比'] = (mzdf['化验费']) / mzdf['费用金额']


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
        mzdf['event费用金额meanlog'] = mzdf['event费用金额mean'].apply(tlog)
        mzdf['event费用金额_maxlog'] = mzdf['event费用金额_max'].apply(tlog)
        del mzdf['费用金额'], mzdf['自费金额'], mzdf['部分自付金额'], mzdf['医保支付金额'], mzdf['event费用金额mean'], mzdf['event费用金额_max']
        print(mzdf.shape)  # (152420, 74)
        mzdf.info()
        """
        """
        mzdf.to_csv('../data/mz_all_claim2.csv', encoding='gbk', index=True)

    if mode == 5:
        """
        feature engineering
        """
        mzdf1 = pd.read_csv('../data/mz_all_event2.csv', encoding='gbk', index_col=0, parse_dates=['出生日期', '就诊结帐费用发生日期'])
        mzdf = mzdf1.copy()
        mzdf['出生月'] = mzdf['出生日期'].dt.month
        mzdf['就诊结帐费用发生月'] = mzdf['就诊结帐费用发生日期'].dt.month
        mzdf['就诊结帐费用发生weekday'] = mzdf['就诊结帐费用发生日期'].dt.weekday
        del mzdf['出生日期']
        del mzdf['就诊结帐费用发生日期']
        del mzdf['ROWNUM']
        # 删除名称 因为有代码
        del mzdf['主被保险人']
        del mzdf['出险人姓名']
        del mzdf['险种名称']
        del mzdf['医院名称']
        del mzdf['费用项目代码']
        del mzdf['疾病名称']
        del mzdf['总案号_分案号']
        del mzdf['收据号']
        del mzdf['医院代码']
        del mzdf['疾病代码']
        del mzdf['主被保险人客户号']
        del mzdf['出险人客户号']
        flist = [
            '中成药费', '中草药', '其他费', '化验费',
            '床位费', '手术费', '护理费', '挂号费',
            '材料费', '检查费', '治疗费', '西药费',
            '诊疗费'
        ]
        for i in flist:
            del mzdf[i]
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
        del mzdf['费用金额'], mzdf['自费金额'], mzdf['部分自付金额'], mzdf['医保支付金额']

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
        # print(mzdf['就诊类型名称'].value_counts())

        mzdf = mzdf[mzdf['就诊类型名称'] != '牙科治疗']
        mzdf = mzdf[mzdf['就诊类型名称'] != '生育']
        print(mzdf.shape)
        mzdf1 = mzdf1.loc[mzdf.index]
        print(mzdf1.shape)
        mzdf1.to_csv('../data/mz_rm_tooth_event2.csv', encoding='gbk', index=True)
        del mzdf1
        mzdf = de.build_one_hot_features(mzdf, ['保单号', '生效日期', '人员属性', '证件类型', '性别', '出险原因', '险种代码', '就诊类型名称'])

        fs = FeatureSelector(data=mzdf)
        fs.identify_missing(missing_threshold=0.6)
        print(fs.record_missing)

        fs.identify_single_unique()
        print(fs.record_single_unique)
        fs.identify_collinear(correlation_threshold=0.975)
        print(fs.record_collinear)
        """
       drop_feature     corr_feature  corr_value
0   生效日期_2016-06-01  保单号_82200946504    1.000000
1   生效日期_2017-06-01  保单号_82200946505    1.000000
2   生效日期_2018-06-01  保单号_82200946506    1.000000
3       人员属性_连带被保险人       人员属性_主被保险人   -1.000000
4            证件类型_8           证件类型_0   -0.995565
5              性别_男             性别_女   -1.000000
6           出险原因_疾病          出险原因_意外   -1.000000
7        险种代码_NIK01       人员属性_主被保险人   -0.991046
8        险种代码_NIK01      人员属性_连带被保险人    0.991046
9        险种代码_NIK02       人员属性_主被保险人    0.988576
10       险种代码_NIK02      人员属性_连带被保险人   -0.988576
11       险种代码_NIK02       险种代码_NIK01   -0.979724
Removed 9 features.
        """
        train_removed = fs.remove(methods=['missing', 'single_unique', 'collinear'])
        train_removed.info()
        """
Int64Index: 267303 entries, 0 to 267302
Data columns (total 38 columns):
年龄                 267303 non-null int64
医院等级               267303 non-null int64
自费总金额占比            267303 non-null float64
医保支付金额占比           267303 non-null float64
药费占比               267303 non-null float64
检查费占比              267303 non-null float64
床位费占比              267303 non-null float64
手术费占比              267303 non-null float64
护理费占比              267303 non-null float64
治疗费占比              267303 non-null float64
诊疗费占比              267303 non-null float64
化验费占比              267303 non-null float64
出生月                267303 non-null int64
就诊结帐费用发生月          267303 non-null int64
就诊结帐费用发生weekday    267303 non-null int64
费用金额log            267303 non-null float64
自费金额log            267303 non-null float64
部分自付金额log          267303 non-null float64
医保支付金额log          267303 non-null float64
保单号_82200946504    267303 non-null uint8
保单号_82200946505    267303 non-null uint8
保单号_82200946506    267303 non-null uint8
人员属性_主被保险人         267303 non-null uint8
证件类型_0             267303 non-null uint8
证件类型_1             267303 non-null uint8
性别_女               267303 non-null uint8
出险原因_意外            267303 non-null uint8
险种代码_MIK01         267303 non-null uint8
险种代码_NIK11         267303 non-null uint8
就诊类型名称_住院就诊        267303 non-null uint8
就诊类型名称_其他约定        267303 non-null uint8
就诊类型名称_接种疫苗        267303 non-null uint8
就诊类型名称_牙科治疗        267303 non-null uint8
就诊类型名称_生育          267303 non-null uint8
就诊类型名称_精神疾病门诊      267303 non-null uint8
就诊类型名称_药房购药        267303 non-null uint8
就诊类型名称_门诊就诊        267303 non-null uint8
就诊类型名称_预防性检查       267303 non-null uint8
dtypes: float64(14), int64(5), uint8(19)
        """
        train_removed.to_csv('../data/mz_rm_tooth_train_data2.csv', encoding='gbk', index=True)

    if mode == 3:
        """
        收据压成事件
            同一时间、同一医院、同一科室（没有科室的话，就同一诊断）
            只要把费用和相对应的就诊类型名称相加即可
        """
        mzdf = pd.read_csv('../data/mz_all_receipt2.csv', encoding='gbk', index_col=0, parse_dates=['出生日期', '就诊结帐费用发生日期'])
        flist = [
            '费用金额', '自费金额', '部分自付金额', '医保支付金额',
            '中成药费', '中草药', '其他费', '化验费',
            '床位费', '手术费', '护理费', '挂号费',
            '材料费', '检查费', '治疗费', '西药费',
            '诊疗费'
        ]
        train_removedg = mzdf.groupby(['总案号_分案号', '医院代码', '疾病代码', '就诊结帐费用发生日期'])[flist].sum()
        mzdf = mzdf.drop_duplicates(subset=['总案号_分案号', '医院代码', '疾病代码', '就诊结帐费用发生日期'], keep='first')
        print(mzdf.shape)  # (267303, 43)
        for i in flist:
            del mzdf[i]
        mzdf = pd.merge(mzdf, train_removedg, how='left', left_on=['总案号_分案号', '医院代码', '疾病代码', '就诊结帐费用发生日期'], right_index=True)

        del mzdf['费用合计']
        flist = [
            '中成药费', '中草药', '其他费', '化验费',
            '床位费', '手术费', '护理费', '挂号费',
            '材料费', '检查费', '治疗费', '西药费',
            '诊疗费'
        ]

        # 构造特征
        # 自费金额占比
        # 自费总金额占费用金额比
        mzdf['自费总金额占比'] = (mzdf['自费金额'] + mzdf['部分自付金额']) / mzdf['费用金额']
        # 医保支付金额占比
        mzdf['医保支付金额占比'] = mzdf['医保支付金额'] / mzdf['费用金额']
        # 费用项目占比
        mzdf['药费占比'] = (mzdf['中成药费'] + mzdf['中草药'] + mzdf['西药费']) / mzdf['费用金额']
        # 检查费占比
        mzdf['检查费占比'] = (mzdf['检查费']) / mzdf['费用金额']
        mzdf['床位费占比'] = (mzdf['床位费']) / mzdf['费用金额']
        mzdf['手术费占比'] = (mzdf['手术费']) / mzdf['费用金额']
        mzdf['护理费占比'] = (mzdf['护理费']) / mzdf['费用金额']
        mzdf['治疗费占比'] = (mzdf['治疗费']) / mzdf['费用金额']
        mzdf['诊疗费占比'] = (mzdf['诊疗费']) / mzdf['费用金额']
        mzdf['化验费占比'] = (mzdf['化验费']) / mzdf['费用金额']

        print(mzdf.shape)  # (267303, 39)
        mzdf.info()
        """

        """
        mzdf['event_id'] = range(mzdf.shape[0])
        mzdf.index = mzdf['event_id']
        del mzdf['event_id']
        mzdf.to_csv('../data/mz_all_event2.csv', encoding='gbk', index=True)
    if mode == 21:

        mzdf = pd.read_csv('../data/mz_all.csv', index_col=0, encoding='gbk', parse_dates=['生效日期', '出生日期', '就诊结帐费用发生日期'])
        mzdf1 = pd.read_csv('../data/mz_all(1).csv', encoding='gbk', parse_dates=['生效日期', '出生日期', '就诊结帐费用发生日期'])
        mzdf['费用项目名称1'] = mzdf1['费用项目名称']
        print( mzdf['费用项目名称1'].unique())
        print( mzdf['费用项目名称'].unique())
        # print(mzdf)
        # print(mzdf1)
        # del mzdf1
        # def t(x):
        #     return ' '.join(set(x.tolist()))
        # print(mzdf.groupby('费用项目名称')['费用项目名称1'].agg(t))

    if mode == 2:
        """
        明细压成收据
        """
        mzdf = pd.read_csv('../data/mz_all.csv', index_col=0, encoding='gbk', parse_dates=['生效日期', '出生日期', '就诊结帐费用发生日期'])

        mzdf['费用项目名称'][mzdf['费用项目名称']=='中成药'] = '中成药费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='麻醉费'] = '治疗费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='留观费'] = '诊疗费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='MRI/CT费'] = '检查费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='CT费'] = '检查费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='B超费'] = '检查费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='合计金额'] = '治疗费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='急救费'] = '治疗费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='专家挂号费'] = '挂号费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='心电费'] = '检查费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='体检费'] = '检查费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='取暖费'] = '其他费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='放射费'] = '检查费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='注射费'] = '治疗费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='服务项目-洗牙'] = '治疗费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='门诊手术费'] = '手术费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='卫材费'] = '材料费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='核磁费'] = '检查费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='空调费'] = '其他费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='病理费'] = '检查费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='氧气费'] = '治疗费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='输氧费'] = '治疗费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='膳食费'] = '其他费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='理疗费'] = '治疗费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='输血费'] = '治疗费'
        mzdf['费用项目名称'][mzdf['费用项目名称']=='接种疫苗费用'] = '治疗费'
        # print(mzdf['费用项目名称'].value_counts())

        flist = [
            '费用金额', '自费金额', '部分自付金额', '医保支付金额'
            ]
        lg1 = ['总案号_分案号', '收据号']
        train_removedg = mzdf.groupby(lg1)[flist].sum()
        mzdf['总案号_分案号_收据号'] = mzdf['总案号_分案号'].apply(lambda x: x+'_') + mzdf['收据号']
        df = mzdf.groupby(['总案号_分案号_收据号', '费用项目名称'])['费用金额'].sum()
        del mzdf['费用金额']
        mzdf = mzdf.drop_duplicates(subset=['总案号_分案号_收据号', '费用项目名称'], keep='first')
        mzdf = pd.merge(mzdf, df, left_on=['总案号_分案号_收据号', '费用项目名称'], right_index=True)
        # mzdf.to_csv('../data/mz_ttt.csv', encoding='gbk', index=True)
        df = mzdf.pivot(index='总案号_分案号_收据号', columns='费用项目名称', values='费用金额').fillna(0)
        # df.to_csv('../data/mz_ttt1.csv', encoding='gbk', index=True)
        del mzdf['费用项目名称']
        mzdf = mzdf.drop_duplicates(subset=lg1, keep='first')
        mzdf = pd.merge(mzdf, df, how='left', left_on='总案号_分案号_收据号', right_index=True)
        mzdf.to_csv('../data/mz_ttt2.csv', encoding='gbk', index=True)
        del mzdf['总案号_分案号_收据号']
        del df
        print(mzdf.shape)#(657371, 43)
        for i in flist:
            del mzdf[i]
        mzdf = pd.merge(mzdf, train_removedg, how='left', left_on=lg1, right_index=True)
        # print(mzdf[abs(mzdf['费用合计']-mzdf['费用金额']) > 0.001][['费用合计', '费用金额']])
        """
           费用合计      费用金额
5081     126.00    297.30
12919    552.50    852.50
22890    200.00    732.00
23552   1826.70   2381.90
23616     18.60   4312.63
23644    168.02    472.66
23950    420.00   1548.70
24497    159.40    325.64
24540    300.00   2510.00
24787   1176.00  12060.00
24797   1176.00  12060.00
24828    779.00   8090.00
24838    779.00   8090.00
28061      8.70    117.40
29094      5.00   2556.23
29443    386.80    394.80
39655    144.10    147.10
42561    189.10   1552.40
48577      1.00      6.00
53232    858.53   1717.06
54836    605.80   1211.60
54921     52.80     72.60
61450      5.00    171.92
66372    204.80    409.60
72074    114.80    230.79
73372    859.98   2238.17
73862     66.00    246.30
90733    118.80    283.80
92308    137.60    144.60
92448    449.00   1032.65
...         ...       ...
231056   346.72    357.74
235005     3.00      4.00
237330   171.31    179.31
237858   278.86    385.76
238168    14.40     30.80
243758   208.36    347.44
243905    16.00     40.00
243910     8.00     32.00
244081   461.35    641.85
246621   470.11    480.71
254348   229.44    260.13
261590    25.44     49.09
262004   337.10    345.10
263051   381.42    389.42
266497     2.00     12.00
268076   213.92    221.92
268081   122.56    267.16
274258    20.00     36.00
274908    84.57     99.97
275282   407.85    415.85
277200   374.00   1496.00
279888    16.71     77.21
286152    25.30    331.62
287785    10.00     20.00
289581    12.00     23.00
290657     8.78     19.38
293802   259.90    299.90
296579    44.00    818.80
297799   668.12    885.00
301146   292.35    348.35

[339 rows x 2 columns]
        """
        # mzdf.info()
        """
Int64Index: 657371 entries, 0 to 301576
Data columns (total 43 columns):
ROWNUM        657371 non-null int64
保单号           657371 non-null int64
生效日期          657371 non-null datetime64[ns]
出险人姓名         657371 non-null object
主被保险人         657371 non-null object
主被保险人客户号      657371 non-null int64
人员属性          657371 non-null object
证件类型          657371 non-null int64
出险人客户号        657371 non-null int64
性别            657371 non-null object
年龄            657371 non-null int64
出生日期          657371 non-null datetime64[ns]
出险原因          657371 non-null object
险种名称          657371 non-null object
险种代码          657371 non-null object
收据号           657371 non-null object
医院代码          657371 non-null object
医院名称          657371 non-null object
医院等级          657371 non-null int64
就诊结帐费用发生日期    657371 non-null datetime64[ns]
费用合计          657371 non-null float64
费用项目代码        657371 non-null object
疾病代码          657371 non-null object
疾病名称          657371 non-null object
就诊类型名称        656883 non-null object
总案号_分案号       657371 non-null object
中成药费          657371 non-null float64
中草药           657371 non-null float64
其他费           657371 non-null float64
化验费           657371 non-null float64
床位费           657371 non-null float64
手术费           657371 non-null float64
护理费           657371 non-null float64
挂号费           657371 non-null float64
材料费           657371 non-null float64
检查费           657371 non-null float64
治疗费           657371 non-null float64
西药费           657371 non-null float64
诊疗费           657371 non-null float64
费用金额          657371 non-null float64
自费金额          657371 non-null float64
部分自付金额        657371 non-null float64
医保支付金额        657371 non-null float64
dtypes: datetime64[ns](3), float64(18), int64(7), object(15)
        """
        mzdf.to_csv('../data/mz_all_receipt2.csv', encoding='gbk', index=True)

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
        print(mzdf['总案号_分案号'].unique().shape)#(152420,)
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
        train_removed['医院等级'] = de.hosRank(train_removed['医院等级'])
        train_removed.info()
        """
Int64Index: 996271 entries, 0 to 301576
Data columns (total 31 columns):
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
医院等级          996271 non-null int64
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
总案号_分案号       996271 non-null object
dtypes: float64(5), int64(7), object(19)      
        """
        train_removed.to_csv('../data/mz_all.csv', encoding='gbk', index=True)
        # train_removed.iloc[:10000].to_csv('../data/mz_all1.csv', encoding='gbk', index=True)
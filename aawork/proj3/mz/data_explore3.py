#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: data_explore3.py
@time: 2019-05-09 19:51

以人为单位进行分析
"""
import pandas as pd
from feature_selector import FeatureSelector


if __name__ == '__main__':
    mode = 2
    if mode == 2:
        """
        featured
        """
        risk_taker_df = pd.read_csv('../data/mz_risk_taker_df.csv', encoding='gbk', index_col=0)
        fs = FeatureSelector(data=risk_taker_df)
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
        """
        """
        train_removed.to_csv('../data/mz_risk_taker_train_data.csv', encoding='gbk', index=True)

    if mode == 1:
        mzdf = pd.read_csv('../data/mz_rm_tooth_event2.csv', encoding='gbk', index_col=0, parse_dates=['出生日期', '就诊结帐费用发生日期'])

        mzdf.index = mzdf['就诊结帐费用发生日期']
        mzdf['就诊结帐费用发生年月'] = mzdf['就诊结帐费用发生日期'].to_period('M').index
        # print(mzdf['就诊结帐费用发生年月'][:5])
        mzdf['就诊结帐费用发生年季度'] = mzdf['就诊结帐费用发生日期'].to_period('Q').index
        # print(mzdf['就诊结帐费用发生年季度'][:5])
        mzdf['就诊结帐费用发生年'] = mzdf['就诊结帐费用发生日期'].to_period('A').index

        mzdf['自费总金额'] = mzdf['自费金额'] + mzdf['部分自付金额']
        del mzdf['自费金额'], mzdf['部分自付金额']
        mzdf['药费'] = mzdf['中成药费'] + mzdf['中草药'] + mzdf['西药费']
        del mzdf['中成药费'], mzdf['中草药'], mzdf['西药费'], mzdf['挂号费']
        # print(mzdf['就诊结帐费用发生年'][:5])
        # print(mzdf['出险人客户号'].unique().shape)#(27463,)
        # print(mzdf['主被保险人客户号'].unique().shape)#(15779,)

        # 统计出险人事件数, 每月/Q去不同医院、得不同诊断数，每月/Q总费用、总自费费用、总医保支付费用 占比
        def t(arr):
            return arr.unique().shape[0]
        risk_taker_df = mzdf.groupby('出险人客户号')['医院代码', '疾病代码'].agg(t)
        risk_taker_df.columns = ['医院代码总uniq个数', '疾病代码总uniq个数']
        risk_taker_df['event_num'] = mzdf.groupby('出险人客户号')['医院代码'].count()
        flist = [
            '费用金额', '自费总金额', '医保支付金额',
            '药费', '其他费', '化验费',
            '床位费', '手术费', '护理费',
            '材料费', '检查费', '治疗费', '诊疗费'
        ]

        for f in flist:
            risk_taker_df[f] = mzdf.groupby('出险人客户号')[f].sum()

        risk_taker_df_per_month = mzdf.groupby(['出险人客户号', '就诊结帐费用发生年月'])['医院代码', '疾病代码'].agg(t)
        risk_taker_df_per_month.columns = ['医院代码_uniqnum_per_month', '疾病代码_uniqnum_per_month']
        risk_taker_df_per_month['event_num_per_month'] = mzdf.groupby(['出险人客户号', '就诊结帐费用发生年月'])['医院代码'].count()
        for f in flist:
            risk_taker_df_per_month[f+'_per_month'] = mzdf.groupby(['出险人客户号', '就诊结帐费用发生年月'])[f].sum()
        for f in flist[1:]:
            risk_taker_df_per_month[f+'_ratio_per_month'] = risk_taker_df_per_month[f+'_per_month'] / risk_taker_df_per_month[flist[0]+'_per_month']
            del risk_taker_df_per_month[f+'_per_month']
        # print(risk_taker_df_per_month.unstack(level=1).fillna(0))
        risk_taker_df = pd.merge(risk_taker_df, risk_taker_df_per_month.unstack(level=1).fillna(0), how='left', left_on='出险人客户号', right_index=True)
        # print(risk_taker_df.info())
        del risk_taker_df_per_month

        risk_taker_df_per_Q = mzdf.groupby(['出险人客户号', '就诊结帐费用发生年季度'])['医院代码', '疾病代码'].agg(t)
        risk_taker_df_per_Q.columns = ['医院代码_uniqnum_per_Q', '疾病代码_uniqnum_per_Q']
        risk_taker_df_per_Q['event_num_per_Q'] = mzdf.groupby(['出险人客户号', '就诊结帐费用发生年季度'])['医院代码'].count()
        for f in flist:
            risk_taker_df_per_Q[f + '_per_Q'] = mzdf.groupby(['出险人客户号', '就诊结帐费用发生年季度'])[f].sum()
        for f in flist[1:]:
            risk_taker_df_per_Q[f+'_ratio_per_Q'] = risk_taker_df_per_Q[f+'_per_Q'] / risk_taker_df_per_Q[flist[0]+'_per_Q']
            del risk_taker_df_per_Q[f+'_per_Q']

        # print(risk_taker_df_per_month.unstack(level=1).fillna(0))
        risk_taker_df = pd.merge(risk_taker_df, risk_taker_df_per_Q.unstack(level=1).fillna(0), how='left',
                                 left_on='出险人客户号', right_index=True)
        # print(risk_taker_df.info())
        del risk_taker_df_per_Q

        risk_taker_df_per_A = mzdf.groupby(['出险人客户号', '就诊结帐费用发生年'])['医院代码', '疾病代码'].agg(t)
        risk_taker_df_per_A.columns = ['医院代码_uniqnum_per_A', '疾病代码_uniqnum_per_A']
        risk_taker_df_per_A['event_num_per_A'] = mzdf.groupby(['出险人客户号', '就诊结帐费用发生年'])['医院代码'].count()
        for f in flist:
            risk_taker_df_per_A[f + '_per_A'] = mzdf.groupby(['出险人客户号', '就诊结帐费用发生年'])[f].sum()
        for f in flist[1:]:
            risk_taker_df_per_A[f+'_ratio_per_A'] = risk_taker_df_per_A[f+'_per_A'] / risk_taker_df_per_A[flist[0]+'_per_A']
            del risk_taker_df_per_A[f+'_per_A']

        # print(risk_taker_df_per_month.unstack(level=1).fillna(0))
        risk_taker_df = pd.merge(risk_taker_df, risk_taker_df_per_A.unstack(level=1).fillna(0), how='left',
                                 left_on='出险人客户号', right_index=True)
        # print(risk_taker_df.info())
        del risk_taker_df_per_A
        print(risk_taker_df.shape)
        risk_taker_df.info()
        risk_taker_df.to_csv('../data/mz_risk_taker_df.csv', encoding='gbk', index=True)
    # print(risk_taker_df_per_month.index['出险人客户号'])
        # print(risk_taker_df_per_month.index['就诊结帐费用发生年月'])
        # for y in [2016, 2017, 2018]:
        #     for m in range(1, 13):
        #         def t(arr):
        #             arr = arr[arr.dt.year == y]
        #             arr = arr[arr.dt.month == m]
        #             return arr.shape[0]
        #         risk_taker_df[f'event_num_{y}_{m}'] = mzdf.groupby('出险人客户号')['就诊结帐费用发生日期'].agg(t)
        #
        #
        #         def t(arr):
        #             arr = arr[arr.dt.year == y]
        #             arr = arr[arr.dt.month == m]
        #             return arr.unique().shape[0]
        #
        #
        #         risk_taker_df = mzdf.groupby('出险人客户号')['医院代码', '疾病代码'].agg(t)
        #         risk_taker_df['event_num'] = mzdf.groupby('出险人客户号')['医院代码'].count()
        # 每月数据

        # print(risk_taker_df.head())
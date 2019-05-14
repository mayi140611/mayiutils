#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: abnormal_detection_gaussian.py
@time: 2019-04-18 18:03
"""
import pandas as pd
from mayiutils.pickle_wrapper import PickleWrapper as picklew
from feature_selector import FeatureSelector


if __name__ == '__main__':
    mode = 4
    if mode == 4:
        """
        feature selector
        """
        # df1 = pd.read_excel('data.xlsx').iloc[:, 1:]
        # print(df1.info())
        df = pd.read_excel('/Users/luoyonggui/Documents/work/dataset/0/data.xlsx')
        # print(df.info())# 查看df字段和缺失值信息
        label = df['理赔结论']
        df = df.drop(columns=['理赔结论'])

        fs = FeatureSelector(data=df, labels=label)
        # 缺失值处理
        fs.identify_missing(missing_threshold=0.6)

    if mode == 3:
        """
        合并参保人基本信息
        """
        df1 = pd.read_excel('data.xlsx', 'Sheet2').dropna(axis=1, how='all')
        # print(df1.info())
        """
归并客户号              528 non-null int64
性别                 528 non-null object
出生年月日              528 non-null datetime64[ns]
婚姻状况               432 non-null object
职业                 484 non-null float64
职业危险等级             484 non-null float64
年收入                528 non-null int64
年交保费               528 non-null float64
犹豫期撤单次数            528 non-null int64
既往理赔次数             528 non-null int64
既往拒保次数             528 non-null int64
既往延期承保次数           528 non-null int64
非标准体承保次数           528 non-null int64
既往调查标识             528 non-null object
既往体检标识             528 non-null object
累积寿险净风险保额          528 non-null float64
累积重疾净风险保额          528 non-null float64
投保人年收入与年交保费比值      437 non-null float64
被保险人有效重疾防癌险保单件数    528 non-null int64
被保险人有效短期意外险保单件数    528 non-null int64
被保险人有效短期健康险保单件数    528 non-null int64
被保险人90天内生效保单件数     528 non-null int64
被保险人180天内生效保单件数    528 non-null int64
被保险人365天内生效保单件数    528 non-null int64
被保险人730天内生效保单件数    528 non-null int64
客户黑名单标识            528 non-null object
保单失效日期             11 non-null datetime64[ns]
保单复效日期             7 non-null datetime64[ns]
受益人变更日期            12 non-null datetime64[ns]
        """
        cols = list(df1.columns)
        cols.remove('保单失效日期')
        cols.remove('保单复效日期')
        cols.remove('受益人变更日期')
        cols.remove('客户黑名单标识')#只有一个值
        df1['出生年'] = df1['出生年月日'].apply(lambda x: int(str(x)[:4]))
        cols.append('出生年')
        cols.remove('出生年月日')
        t = pd.get_dummies(df1['婚姻状况'], prefix='婚姻状况')
        df2 = pd.concat([df1, t], axis=1)
        print(df2.shape)
        cols.extend(list(t.columns))
        cols.remove('婚姻状况')
        t = pd.get_dummies(df2['性别'], prefix='性别')
        df2 = pd.concat([df2, t], axis=1)
        print(df2.shape)
        cols.extend(list(t.columns))
        cols.remove('性别')
        t = pd.get_dummies(df2['既往调查标识'], prefix='既往调查标识')
        df2 = pd.concat([df2, t], axis=1)
        print(df2.shape)
        cols.extend(list(t.columns))
        cols.remove('既往调查标识')
        t = pd.get_dummies(df2['既往体检标识'], prefix='既往体检标识')
        df2 = pd.concat([df2, t], axis=1)
        print(df2.shape)
        cols.extend(list(t.columns))
        cols.remove('既往体检标识')
        # print(df2['职业'].value_counts())
        """
        取前四位 分类
        """
        df2['职业'] = df2['职业'].apply(lambda x: str(x)[:1])
        # print(df2['职业'].value_counts())
        t = pd.get_dummies(df2['职业'], prefix='职业')
        df2 = pd.concat([df2, t], axis=1)
        print(df2.shape)
        cols.extend(list(t.columns))
        cols.remove('职业')
        print(df2['职业危险等级'].value_counts())
        t = pd.get_dummies(df2['职业危险等级'], prefix='职业危险等级')
        df2 = pd.concat([df2, t], axis=1)
        print(df2.shape)
        cols.extend(list(t.columns))
        cols.remove('职业危险等级')
        df2['投保人年收入与年交保费比值'] = df2['投保人年收入与年交保费比值'].fillna(0)
        """
        归并客户号有重复的，取重复值的第一条
        """
        df2 = df2.drop_duplicates(subset=['归并客户号'], keep='first')
        print(df2['归并客户号'].value_counts())
        df2 = df2.rename(columns={'归并客户号': '被保人归并客户号'})
        cols.remove('归并客户号')
        cols.append('被保人归并客户号')
        # print(df2[cols].info())

        #合并
        train_df = picklew.loadFromFile('train_data1.pkl')

        print(train_df.shape)
        train_df = pd.merge(train_df, df2[cols], how='left', on='被保人归并客户号')
        del train_df['营销员工号']
        del train_df['被保人核心客户号']
        del train_df['保人归并客户号']
        del train_df['被保人归并客户号']
        print(train_df.shape)  # (562, 30)
        print(train_df.info())

        del train_df['理赔金额']
        picklew.dump2File(train_df, 'train_data2.pkl')

    if mode == 2:
        """
        合并销售人员信息
        """
        df1 = pd.read_excel('data.xlsx', 'Sheet1')
        # print(df1.info())
        """
RangeIndex: 532 entries, 0 to 531
Data columns (total 7 columns):
营销员工号           532 non-null int64
营销员黑名单标记        326 non-null object
营销员入司时间         326 non-null datetime64[ns]
营销员离职时间         97 non-null datetime64[ns]
营销员所售保单数量       532 non-null int64
营销员所售保单标准体数量    532 non-null int64
营销员所售保单出险数量     532 non-null int64
        """
        cols = list(df1.columns)
        # print(df1['营销员黑名单标记'].value_counts())
        """
        全部都是N, 没有意义， 删除
        """
        cols.remove('营销员离职时间')
        df2 = df1[cols].dropna()
        cols.remove('营销员黑名单标记')
        cols.remove('营销员入司时间')
        df2 = df2[cols]
        # print(df2.info())
        """
营销员工号           326 non-null int64
营销员所售保单数量       326 non-null int64
营销员所售保单标准体数量    326 non-null int64
营销员所售保单出险数量     326 non-null int64
        """
        # print(df2['营销员工号'].value_counts())
        # print(df2.info())
        #合并df
        train_df = picklew.loadFromFile('train_data.pkl')
        train_df = train_df.rename(columns={'(SELECTDISTINCTLJ.AGENTCODEFRO销售人员工号':'营销员工号'})
        print(train_df.shape)
        # train_df = pd.merge(train_df, df2, how='left', on='营销员工号')
        print(train_df.shape)#(562, 30)
        print(train_df.info())
        picklew.dump2File(train_df, 'train_data1.pkl')

    if mode == 1:
        """
        主表
        """
        df1 = pd.read_excel('data.xlsx').iloc[:, 1:]
        # print(df1.shape)#(562, 41)
        # print(df1.columns)
        """
        ['平台流水号', '保单管理机构', '保单号', '指定受益人标识', '受益人与被保险人关系', '交费方式',
           '交费期限', '核保标识', '核保结论', '投保时年龄', '基本保额与体检保额起点比例', '生调保额起点',
           '投保保额临近核保体检临界点标识', '投保保额', '临近核保生调临界点标识', '理赔金额', '累计已交保费', '理赔结论',
           'Unnamed: 19', '生效日期', '出险前最后一次复效日期', '承保后最小借款日期', '出险日期', '报案时间',
           '申请日期', '出险减生效天数', '出险减最后一次复效天数', '重疾保单借款减生效日期天数', '申请时间减出险时间',
           '报案时间减出险时间', '出险原因1', '出险原因2', '出险原因3', '出险结果', '保单借款展期未还次数', '失复效记录次数',
           '销售渠道', '(SELECTDISTINCTLJ.AGENTCODEFRO销售人员工号', '被保人核心客户号', '保人归并客户号',
           '被保人归并客户号']
        """
        # 删除全部为null的列
        df2 = df1.dropna(axis=1, how='all')
        # print(df2.shape)#(562, 33)
        # print(df2.columns)
        """
['平台流水号', '保单管理机构', '保单号', '指定受益人标识', '受益人与被保险人关系', '交费方式', '交费期限',
       '核保标识', '核保结论', '投保时年龄', '投保保额', '理赔金额', '累计已交保费', '理赔结论',
       'Unnamed: 19', '生效日期', '出险前最后一次复效日期', '承保后最小借款日期', '出险日期', '报案时间',
       '申请日期', '出险减生效天数', '出险减最后一次复效天数', '申请时间减出险时间', '报案时间减出险时间', '出险原因1',
       '出险结果', '失复效记录次数', '销售渠道', '(SELECTDISTINCTLJ.AGENTCODEFRO销售人员工号',
       '被保人核心客户号', '保人归并客户号', '被保人归并客户号']     
        """
        # print(df2.info())
        """
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
投保保额                                    562 non-null float64
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
申请时间减出险时间                               562 non-null int64
报案时间减出险时间                               119 non-null float64
出险原因1                                   562 non-null object
出险结果                                    552 non-null object
失复效记录次数                                 562 non-null int64
销售渠道                                    562 non-null object
(SELECTDISTINCTLJ.AGENTCODEFRO销售人员工号    562 non-null int64
被保人核心客户号                                562 non-null int64
保人归并客户号                                 562 non-null int64
被保人归并客户号                                562 non-null int64
        """

        train_col = list(df2.columns)

        train_col.remove('平台流水号')
        train_col.remove('Unnamed: 19')
        train_col.remove('生效日期')
        train_col.remove('出险日期')
        train_col.remove('报案时间')
        train_col.remove('申请日期')
        train_col.remove('出险减最后一次复效天数')
        train_col.remove('报案时间减出险时间')
        train_col.remove('出险前最后一次复效日期')
        train_col.remove('承保后最小借款日期')
        # print(df2[train_col].info())
        """
保单管理机构                                  562 non-null int64
保单号                                     562 non-null int64
指定受益人标识                                 562 non-null object
受益人与被保险人关系                              538 non-null object
交费方式                                    562 non-null object
交费期限                                    562 non-null int64
核保标识                                    562 non-null object
核保结论                                    544 non-null object
投保时年龄                                   562 non-null int64
投保保额                                    562 non-null float64
理赔金额                                    562 non-null float64
累计已交保费                                  562 non-null float64
出险减生效天数                                 562 non-null int64
申请时间减出险时间                               562 non-null int64
出险原因1                                   562 non-null object
出险结果                                    552 non-null object
失复效记录次数                                 562 non-null int64
销售渠道                                    562 non-null object
(SELECTDISTINCTLJ.AGENTCODEFRO销售人员工号    562 non-null int64
被保人核心客户号                                562 non-null int64
保人归并客户号                                 562 non-null int64
被保人归并客户号                                562 non-null int64       
        """

        label = df2['理赔结论']
        train_col.remove('理赔结论')#删除label

        # print(label.value_counts())
        """
        正常给付    432
        全部拒付    107
        协议给付     15
        部分给付      8
        """
        # print(df1['保单号'].value_counts())
        train_col.remove('保单号')
        # print(df1['保单管理机构'].value_counts())
        """
        取前4位
        """
        df2['保单管理机构'] = df2['保单管理机构'].copy().apply(lambda x: str(x)[:4])
        # print(df2['保单管理机构'].value_counts())
        """
8603    280
8602    163
8605     65
8604     34
8606     16
8608      4
        """
        t = pd.get_dummies(df2['指定受益人标识'], prefix='指定受益人标识')
        print(df2.shape)
        df2 = pd.concat([df2, t], axis=1)
        print(df2.shape)
        train_col.extend(list(t.columns))
        train_col.remove('指定受益人标识')
        t = pd.get_dummies(df2['交费方式'], prefix='交费方式')
        print(df2.shape)
        df2 = pd.concat([df2, t], axis=1)
        print(df2.shape)
        train_col.extend(list(t.columns))
        train_col.remove('交费方式')
        t = pd.get_dummies(df2['核保标识'], prefix='核保标识')
        print(df2.shape)
        df2 = pd.concat([df2, t], axis=1)
        print(df2.shape)
        train_col.extend(list(t.columns))
        train_col.remove('核保标识')
        t = pd.get_dummies(df2['保单管理机构'], prefix='保单管理机构')
        print(df2.shape)
        df2 = pd.concat([df2, t], axis=1)
        print(df2.shape)
        train_col.extend(list(t.columns))
        train_col.remove('保单管理机构')
        t = pd.get_dummies(df2['出险原因1'], prefix='出险原因1')
        print(df2.shape)
        df2 = pd.concat([df2, t], axis=1)
        print(df2.shape)
        train_col.extend(list(t.columns))
        train_col.remove('出险原因1')
        t = pd.get_dummies(df2['销售渠道'], prefix='销售渠道')
        print(df2.shape)
        df2 = pd.concat([df2, t], axis=1)
        print(df2.shape)
        train_col.extend(list(t.columns))
        train_col.remove('销售渠道')
        # print(df2['受益人与被保险人关系'].value_counts())
        """
 本人    530
父亲      2
丈夫      2
配偶      1
其他      1
父母      1
母亲      1       

父亲 母亲 合并为父母； 丈夫 合并到 配偶；
        """
        t = df2['受益人与被保险人关系'].copy()
        t[t == '母亲'] = '父母'
        t[t == '父亲'] = '父母'
        t[t == '丈夫'] = '配偶'
        df2['受益人与被保险人关系'] = t
        print(df2['受益人与被保险人关系'].value_counts())
        t = pd.get_dummies(df2['受益人与被保险人关系'], prefix='受益人与被保险人关系')
        print(df2.shape)
        df2 = pd.concat([df2, t], axis=1)
        print(df2.shape)
        train_col.extend(list(t.columns))
        train_col.remove('受益人与被保险人关系')
        # print(df2['受益人与被保险人关系'].value_counts())
        # print(df2['核保结论'].value_counts())
        train_col.remove('核保结论')
        """
        后验数据，删除
        """
        # print(df2['出险结果'].value_counts())
        """
        感觉也可以处理一下，比如把肿瘤的标出来，
        暂时先删掉吧
        """
        train_col.remove('出险结果')
        print(len(train_col))
        print(df2[train_col].info())
        picklew.dump2File(df2[train_col], 'train_data.pkl')
        label[label != '正常给付'] = int(1)
        label[label == '正常给付'] = int(0)
        print(label.value_counts())
        picklew.dump2File(label, 'label.pkl')


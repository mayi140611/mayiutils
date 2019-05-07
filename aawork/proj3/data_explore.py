#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: data_explore.py
@time: 2019-05-06 17:22
"""
import pandas as pd


if __name__ == '__main__':
    mode = 2
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
        # 构造特征：住院天数
        dfzy['住院天数'] = [t.days+1 for t in (dfzy['出院时间'] - dfzy['入院时间'])]
        # print(dfzy['医保支付描述'].value_counts())
        # print(dfzy['部分自付描述'].value_counts())
        # print(dfzy['自费描述'].value_counts())
        print(dfzy['性别'].value_counts())
        print(dfzy['收据号'].value_counts())
        # print(dfzy['住院天数'])
        # dfzy.info()
        # dfzy.to_csv('zy_all_featured.csv', encoding='gbk', index=False)
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
#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: pandastest.py
@time: 2019/1/26 13:00
"""
import pandas as pd
from mayiutils.db.pymongo_wrapper import PyMongoWrapper
from mayiutils.db.pymysql_wrapper import PyMysqlWrapper
from mayiutils.pickle_wrapper import PickleWrapper as pkw

if __name__=='__main__':
    df = pd.read_excel('tmp/target.xlsx')
    df['new'] = ''

    set1 = set(df.loc[:, '药品通用名称'])
    # for i in set1:
    print(len(set1))
    pmw = PyMongoWrapper('h1')
    dbname = 'yyw'
    tablename = 'drugdetail'
    table = pmw.getCollection(dbname, tablename)
    set2 = set([i['通用名称'] for i in pmw.findAll(table, fieldlist=['通用名称'])])
    print(len(set2))

    ii = set1.intersection(set2)
    print(len(ii))


    pmw = PyMysqlWrapper(host='10.1.192.118')
    pmw.execute('SELECT s.TY_NAME FROM sms_main s ')
    data = pmw._cursor.fetchall()
    data1 = [i[0] for i in data]
    set3 = set(data1)
    print(len(set1.intersection(set3)))
    print(len(set2.intersection(set3)))
    print(len(ii.intersection(set3)))

    print(len(set1.difference(set3).intersection(set2)))
    print(len(set1.difference(set3).difference(set2)))
    # print(df.head())
    # df1 = df.iloc[:, :2].fillna('')
    # d = dict()
    # for line in df1.itertuples():
    #     if line[2]:
    #         d[line[1]] = line[2]
    #     else:
    #         d[line[1]] = line[1]
    #     # print(d)
    # pkw.dump2File(d, 'illsvalid.pkl')
    # print(df1.iloc[:5])
    # # df2 = df1[df1.iloc[:, 0].isin(['偏头痛','脑萎缩'])]
    # df2 = df1[df1.iloc[:, 0].isin(d.keys())]
    # df2['a'] = ''
    # def t(x):
    #     x['a'] = x.iloc[0]
    #     return x
    # print(df2.iloc[:5])
    # df2.apply(t, axis=1)
    # df2['b'] = 1
    # g = df2[['b','a']].groupby(['a']).mean()
    # g1 = g.sort_values('b')
    # print(g1.iloc[:3, 0])

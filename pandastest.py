#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: pandastest.py
@time: 2019/1/26 13:00
"""
import pandas as pd
from mayiutils.pickle_wrapper import PickleWrapper as pkw

if __name__=='__main__':
    df = pd.read_excel('疾病列表190126.xlsx','1281')
    df1 = df.iloc[:, :2].fillna('')
    d = dict()
    for line in df1.itertuples():
        if line[2]:
            d[line[1]] = line[2]
        else:
            d[line[1]] = line[1]
        # print(d)
    pkw.dump2File(d, 'illsvalid.pkl')
    print(df1.iloc[:5])
    # df2 = df1[df1.iloc[:, 0].isin(['偏头痛','脑萎缩'])]
    df2 = df1[df1.iloc[:, 0].isin(d.keys())]
    df2['a'] = ''
    def t(x):
        x['a'] = x.iloc[0]
        return x
    print(df2.iloc[:5])
    df2.apply(t, axis=1)
    df2['b'] = 1
    g = df2[['b','a']].groupby(['a']).mean()
    g1 = g.sort_values('b')
    print(g1.iloc[:3, 0])

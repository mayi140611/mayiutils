#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: test.py
@time: 2019-05-15 15:09
"""
import pandas as pd


if __name__ == '__main__':
    mode = 1
    if mode == 1:
        df = pd.read_excel('zy_all.xlsx', converters={'出险人客户号': str})
        df1 = pd.read_csv('../data/zy_all.csv')
        df1['出险人客户号_完整'] = df['出险人客户号']
        df1.to_excel('zy_all_t.xlsx')
    if mode == 0:
        df6 = pd.read_excel('/Users/luoyonggui/Documents/datasets/work/3/82200946506.xlsx', converters={'出险人客户号': str})
        dfzy = df6[df6['就诊类型'] == '住院']
        df6 = pd.read_excel('/Users/luoyonggui/Documents/datasets/work/3/82200946505.xlsx', converters={'出险人客户号': str})
        dfzy5 = df6[df6['就诊类型'] == '住院']
        df6 = pd.read_excel('/Users/luoyonggui/Documents/datasets/work/3/82200946504.xlsx', converters={'出险人客户号': str})
        dfzy4 = df6[df6['就诊类型'] == '住院']
        dfzy = pd.concat([dfzy4, dfzy5, dfzy])
        dfzy.to_excel('zy_all.xlsx')
        # print(df6['出险人客户号'][:20])
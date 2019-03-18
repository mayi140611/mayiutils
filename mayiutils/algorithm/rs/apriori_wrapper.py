#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: apriori_wrapper.py
@time: 2019/3/17 9:55

https://kexue.fm/archives/3380
Apriori算法是一个寻找关联规则的算法，也就是从一大批数据中找到可能的逻辑，
比如“条件A+条件B”很有可能推出“条件C”（A+B-->C），这就是一个关联规则。
具体来讲，比如客户买了A商品后，往往会买B商品（反之，买了B商品不一定会买A商品），
或者更复杂的，买了A、B两种商品的客户，很有可能会再买C商品（反之也不一定）。
有了这些信息，我们就可以把一些商品组合销售，以获得更高的收益。
而寻求关联规则的算法，就是关联分析算法。
"""
import pandas as pd
import time


def connect_string(x, ms):
    """
    # 自定义连接函数，用于实现L_{k-1}到C_k的连接
    :param x: 带连接的list，如['a', 'b', 'c', 'd', 'e']
    :param ms: 连接符，如'-'
    :return:
        [['a', 'b'], ['a', 'c'], ['a', 'd'], ['a', 'e'], ['b', 'c'],
        ['b', 'd'], ['b', 'e'], ['c', 'd'], ['c', 'e'], ['d', 'e']]
    """
    x = list(map(lambda i: sorted(i.split(ms)), x))
    l = len(x[0])
    r = []
    for i in range(len(x)):
        for j in range(i, len(x)):
            if x[i][:l - 1] == x[j][:l - 1] and x[i][l - 1] != x[j][l - 1]:
                r.append(x[i][:l - 1] + sorted([x[j][l - 1], x[i][l - 1]]))
    return r


def find_rule(d, support, confidence):
    """
    # 寻找关联规则的函数
    :param d:
    :param support:
    :param confidence:
    :return:
    """
    start = time.clock()
    result = pd.DataFrame(index=['support', 'confidence'])  # 定义输出结果

    # support_series = 1.0 * d.sum() / len(d)  # 支持度序列
    support_series = 1.0 * d.sum(axis=0) / d.shape[0]  # 支持度序列
    column = list(support_series[support_series > support].index)  # 初步根据支持度筛选
    print('数目：%s...' % len(column))
    k = 0

    while len(column) > 1:
        k = k + 1
        print('\n正在进行第%s次搜索...' % k)
        column = connect_string(column, ms)
        print('数目：%s...' % len(column))
        sf = lambda i: d[i].prod(axis=1, numeric_only=True)  # 新一批支持度的计算函数

        # 创建连接数据，这一步耗时、耗内存最严重。当数据集较大时，可以考虑并行运算优化。
        d_2 = pd.DataFrame(list(map(sf, column)), index=[ms.join(i) for i in column]).T
        print(d_2.head())
        support_series_2 = 1.0 * d_2[[ms.join(i) for i in column]].sum() / len(d)  # 计算连接后的支持度
        column = list(support_series_2[support_series_2 > support].index)  # 新一轮支持度筛选
        support_series = support_series.append(support_series_2)
        column2 = []

        for i in column:  # 遍历可能的推理，如{A,B,C}究竟是A+B-->C还是B+C-->A还是C+A-->B？
            i = i.split(ms)
            for j in range(len(i)):
                column2.append(i[:j] + i[j + 1:] + i[j:j + 1])

        cofidence_series = pd.Series(index=[ms.join(i) for i in column2])  # 定义置信度序列

        for i in column2:  # 计算置信度序列
            cofidence_series[ms.join(i)] = support_series[ms.join(sorted(i))] / support_series[ms.join(i[:len(i) - 1])]

        for i in cofidence_series[cofidence_series > confidence].index:  # 置信度筛选
            result[i] = 0.0
            result[i]['confidence'] = cofidence_series[i]
            result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]

    result = result.T.sort_values(['confidence', 'support'], ascending=False)  # 结果整理，输出
    end = time.clock()
    print('\n搜索完成，用时：%0.2f秒' % (end - start))
    print('\n结果为：')
    print(result)

    return result


if __name__ == '__main__':
    # print(connect_string(['a', 'b', 'c', 'd', 'e'], '-'))
    d = pd.read_csv('apriori.txt', header=None)
    print(d.head())

    print('\n转换原始数据至0-1矩阵...')

    start = time.clock()
    ct = lambda x: pd.Series(1, index=x)
    b = map(ct, d.values)
    d = pd.DataFrame(list(b)).fillna(0)
    d = (d == 1)
    end = time.clock()
    print('\n转换完毕，用时：%0.2f秒' % (end - start))
    print('\n开始搜索关联规则...')
    del b

    support = 0.06  # 最小支持度
    confidence = 0.75  # 最小置信度
    ms = '--'  # 连接符，用来区分不同元素，如A--B。需要保证原始表格中不含有该字符

    find_rule(d, support, confidence).to_excel('rules.xls')
    """
                     support  confidence
    A3--F4--H4      0.078495    0.879518
    C3--F4--H4      0.075269    0.875000
    B2--F4--H4      0.062366    0.794521
    C2--E3--D2      0.092473    0.754386
    D2--F3--H4--A2  0.062366    0.753247
    """
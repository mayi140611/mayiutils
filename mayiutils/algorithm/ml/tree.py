#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: tree.py
@time: 2019/3/18 10:07

决策树有两个基本问题
    1、如何决定选择哪个变量进行分裂
        方法：测量不纯度，选择不纯度越低的变量优先分裂。
        样本分类越平均，不纯度越大
        测量不纯度的指标：gini系数，熵，分类误差
"""
import pandas as pd
import numpy as np


def calGini(df, featureColName, classColName):
    '''
    计算某个特征的Gini值，注意：该特征必须是离散值
    featureColName： 要计算的特征列的名称
    classColName：分类列的名称
    '''
    classes = df[classColName].unique()
    g_list = list()
    n = df.shape[0]#样本总数
    for i in df[featureColName].unique():
        n1 = df[df[featureColName] == i].shape[0]
        gini = 1
        # 计算每一个feature的Gini值
        for c in classes:
            n2 = df[df[featureColName] == i][df[classColName] == c].shape[0]
            gini = gini - (n2/n1)**2
        gini_w = gini * n1/n # 加权后的基尼系数
        # print('特征：{}\tGini值为{},加权后的Gini值为{}'.format(i,gini,gini_w))
        g_list.append(gini_w)
    giniF = sum(g_list)
    print('特征：{}\tGini值为{}'.format(featureColName, giniF))
    return featureColName, giniF


def selectFeature(df, classColName):
    """
    选择gini值最小的作为分裂特征
    :param df:
    :param classColName:
    :return:
    """
    columns = list(df.columns)
    columns.remove(classColName)
    feature = -1
    gini = 1
    for c in columns:
        f, g = calGini(df, c, classColName)
        if g < gini:
            gini = g
            feature = c
    return feature, gini


if __name__ == '__main__':
    genderdict = {0: '男', 1: '女'}
    typedict = {0: '家用', 1: '运动', 2: '豪华'}
    sizedict = {0: '小', 1: '中', 2: '大', 3: '加大'}
    classdict = {0: 'C0', 1: 'C1'}
    genderlist = [0] * 6 + [1] * 4 + [0] * 4 + [1] * 6
    typelist = [0] + [1] * 8 + [2] + [0] * 3 + [2] * 7
    sizelist = [0] + [1] * 2 + [2] + [3] * 2 + [0] * 2 + [1] + [2] * 2 + [3] + [1] + [3] + [0] * 2 + [1] * 3 + [2]
    classlist = [0] * 10 + [1] * 10
    df = pd.DataFrame({'性别': [genderdict[i] for i in genderlist],
                       '车型': [typedict[i] for i in typelist],
                       '衬衣尺寸': [sizedict[i] for i in sizelist],
                       '类': [classdict[i] for i in classlist]})
    print(df.head())
    # calGini(df, '性别', '类')
    print(selectFeature(df, '类'))
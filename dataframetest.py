#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: dataframetest.py
@time: 2019/1/27 20:58
"""
from mayiutils.pandas.dataframe_wrapper import DataframeWrapper
import pandas as pd

if __name__ == '__main__':
    dfw = DataframeWrapper()
    df =  pd.read_excel("疾病列表190126.xlsx")
    #获取满足要求的样本
    df1 = dfw.isIn(df, '疾病名称（39）', ['嗜睡症', '脑萎缩'])
    print(df1)
#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: data_explore.py
@time: 2019/4/14 10:29
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    mode = 3
    df = pd.read_csv('../../tmp/creditcard.csv')
    # print(df.shape)#(284807, 31)
    # print(df['Class'].value_counts())
    """
    0    284315
    1       492
    """
    # print('ratio:{:.2f}'.format(284315/492))#ratio:577.88
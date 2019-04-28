#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: abnormal_detection_gaussian.py
@time: 2019/3/6 11:03

主要功能：
探索样本的分布、缺失值、取值范围等，
如果有缺失值，进行填补
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


class DataExplore:

    def __init__(self, baseDir='/home/ian/datafoundation'):
        self._baseDir = baseDir

    def countN2Csv(self):
        """
        统计每个肝癌样本的切片数，存入csv
        :return:
        """

        for csv in ['train1_label.csv', 'train2_label.csv']:
            df = pd.read_csv(os.path.join(self._baseDir, csv))
            countlist = list()
            for line in df.itertuples():
                imgdir = os.path.join(self._baseDir, csv.split('_')[0]+'_jpg', line[1])
                count = len(os.listdir(imgdir))
                countlist.append(count)
            df['slicenum'] = countlist
            df.to_csv('{}_slicenum.csv'.format(csv.split('.')[0]))
    def hist(self):
        """
        根据频率直方图的显示结果，选择splicenum为192
        :return:
        """
        df = pd.DataFrame()
        for csv in ['train1_label_slicenum.csv', 'train2_label_slicenum.csv']:
            if df.shape[0] == 0:
                df = pd.read_csv(os.path.join(self._baseDir, csv))
            else:
                df = pd.concat([df, pd.read_csv(os.path.join(self._baseDir, csv))])

        print(df.shape)#(7574, 4)
        print(df['ret'].value_counts())
        """
        0    4278
        1    3296
        """
        plt.figure()
        plt.subplot(221)
        plt.hist(df['slicenum'], bins=50)
        plt.subplot(222)
        plt.hist(df['slicenum'][df['ret']==0], bins=50)
        plt.subplot(223)
        plt.hist(df['slicenum'][df['ret']==1], bins=50)
        plt.show()


if __name__ == '__main__':
    de = DataExplore()
    path = 'D:/Desktop/DF'
    de = DataExplore(baseDir=path)
    de.hist()




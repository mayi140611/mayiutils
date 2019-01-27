#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: datasets_wrapper.py
@time: 2019/1/26 10:32
"""
from sklearn import datasets


class DataSetsWrapper:
    @classmethod
    def loadDiabetes(cls):
        """
        Diabetes(糖尿病)数据集，可用于做线性回归
        包含442个样本，每个样本有10个特征：
            age, sex, body mass index, average blood pressure, and six blood serum（血清） measurements
        Target: Column 11 is a quantitative（定量） measure of disease progression one year after baseline
        :return:
        """
        return datasets.load_diabetes()
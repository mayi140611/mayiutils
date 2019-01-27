#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: sklearntest.py
@time: 2019/1/26 11:28
"""
from mayiutils.sklearn.metrics_wrapper import MetricsWrapper as mw
import numpy as np

if __name__ == '__main__':
    # r = mw.calMeanSquaredError(np.array([1,2,3]), np.array([1,2,3]))
    r = mw.calMeanSquaredError([1,2,3], [1,3,2], multioutput='raw_values')
    r1 = mw.calMeanSquaredError([1,2,3], [1,3,2])
    print(r, r1)

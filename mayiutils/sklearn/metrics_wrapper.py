#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: metrics_wrapper.py
@time: 2019/1/26 11:23
"""
from sklearn.metrics import mean_squared_error, r2_score


class MetricsWrapper:
    """
    模型效果评估
    sklearn中的回归器性能评估方法：
        http://www.cnblogs.com/nolonely/p/7009001.html
        mean_squared_error、
        mean_absolute_error、
        explained_variance_score
        r2_score：
    """
    @classmethod
    def calMeanSquaredError(cls, y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
        """
        计算均方误差
        :param y_true:
        :param y_pred:
        :param sample_weight:
        :param multioutput:
            string in ['raw_values', 'uniform_average']
        :return:
        """
        return mean_squared_error(y_true, y_pred, sample_weight, multioutput)

    @classmethod
    def calR2(cls):
        """
        决定系数（拟合优度）。1表示最好，0表示最差
        R^2 (coefficient of determination) regression score function.

        Best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.
        :return:
        """







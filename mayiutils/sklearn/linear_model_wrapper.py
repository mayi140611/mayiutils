#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: linear_model_wrapper.py
@time: 2019/1/26 11:16
"""
from sklearn.linear_model import LinearRegression

class LinearModelWrapper:

    def linearRegression(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=1):
        """
        Ordinary least squares Linear Regression.
        训练分为3步：
            构建模型
            训练模型fit
            预测模型predict
            # Create linear regression object
            regr = LinearRegression()

            # Train the model using the training sets
            regr.fit(diabetes_X_train, diabetes_y_train)

            # Make predictions using the testing set
            diabetes_y_pred = regr.predict(diabetes_X_test)
        :param fit_intercept:
        :param normalize:
        :param copy_X:
        :param n_jobs:
        :return:
        """
        return LinearRegression(fit_intercept, normalize, copy_X, n_jobs)
    @classmethod
    def logisticRegression(cls):
        """
        逻辑回归模型。
        :return:
        """


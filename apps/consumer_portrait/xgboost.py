#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: xgboost.py
@time: 2019/3/19 7:01
"""
import xgboost
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import os
import math
import itertools


if __name__ == '__main__':
    mode = 2
    baseDir = 'D:/Desktop/DF/portrait'
    df = pd.read_csv(os.path.join(baseDir, 'train_dataset.csv'))
    testdf = pd.read_csv(os.path.join(baseDir, 'test_dataset.csv'))

    print(df.shape)
    print(testdf.shape)

    X = df.iloc[:, 1:29].values
    y = df.iloc[:, 29].values
    if mode == 2:
        """
        采用全数据集进行训练+测试
        """
        l = 0.12
        n = 130
        m = 5
        model = xgboost.XGBRegressor(learning_rate=l, n_estimators=n, max_depth=m)
        model.fit(X, y)
        # print(model.max_depth)
        prediction = model.predict(testdf.iloc[:, 1:].values)
        testdf['prediction'] = np.round(prediction)
        testdf.iloc[:, [0, -1]].to_csv(os.path.join(baseDir, 'test_dataset4.csv'))
    if mode == 1:
        """
        利用网格搜索选择参数
        第一轮参数调优的最优结果0.12 130 5：3.858105129941366 0.7911153342233519 耗时29.07000000000005s
        """
        skf = StratifiedKFold(n_splits=5)
        # lr = [0.05, 0.1, 0.2]
        # max_depth = [3, 5, 7]
        # n_estimators = [80, 100, 120]
        lr = [0.1, 0.12]
        max_depth = [5, 6, 7]
        n_estimators = [110, 120, 130]
        for l, n, m in itertools.product(lr, n_estimators, max_depth):
            print(l, n, m)
            maelist = []
            r2list = []
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = xgboost.XGBRegressor(learning_rate=l, n_estimators=n, max_depth=m)
                model.fit(X_train, y_train)
                prediction = model.predict(X_test)
                mae = math.sqrt(mean_absolute_error(y_test, prediction))
                maelist.append(mae)
                r2list.append(r2_score(y_test, prediction))
            print(np.mean(maelist), np.mean(r2list))
    if mode == 0:
        """
        使用默认参数跑一遍
        3.8992280950972003 0.7816561825408523
        """
        skf = StratifiedKFold(n_splits=5)
        skf.get_n_splits(X, y)
        maelist = []
        r2list = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = xgboost.XGBRegressor()
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            mae = math.sqrt(mean_absolute_error(y_test, prediction))
            maelist.append(mae)
            r2list.append(r2_score(y_test, prediction))
        print(np.mean(maelist), np.mean(r2list))
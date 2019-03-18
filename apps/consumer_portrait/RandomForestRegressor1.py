#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: RandomForestRegressor1.py
@time: 2019/3/19 4:01
"""
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
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
        n = 14
        m = 12
        model = RandomForestRegressor(n_estimators=n, max_depth=m)
        model.fit(X, y)
        # print(model.max_depth)
        prediction = model.predict(testdf.iloc[:, 1:].values)
        testdf['prediction'] = np.round(prediction)
        testdf.iloc[:, [0, -1]].to_csv(os.path.join(baseDir, 'test_dataset3.csv'))
    if mode == 1:
        """
        利用网格搜索选择参数
        得到n_estimators = 14, max_depth = 12时最优 r2值0.7698366625012384
        """
        skf = StratifiedKFold(n_splits=5)
        skf.get_n_splits(X, y)
        # max_depth = [4, 6, 8, 10, 12]
        # max_depth = [6, 8, 10]
        max_depth = [10, 12, 14]
        # n_estimators = [6, 8, 10]
        n_estimators = [10, 12, 14]
        for n, m in itertools.product(n_estimators, max_depth):
            print(n, m)
            rmselist = list()
            r2list = list()
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = RandomForestRegressor(n_estimators=n, max_depth=m)
                model.fit(X_train, y_train)
                # print(model.max_depth)
                prediction = model.predict(X_test)
                # print("accuracy score: ")
                # print(accuracy_score(y_test, prediction))
                # print(classification_report(y_test, prediction))
                rmse = math.sqrt(mean_squared_error(y_test, prediction))
                rmselist.append(rmse)
                r2list.append(r2_score(y_test, prediction))
            # print(np.mean(rmselist), np.mean(r2list))
            print(np.mean(r2list))
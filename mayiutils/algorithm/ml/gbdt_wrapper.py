#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: gbdt_wrapper.py
@time: 2019-04-24 12:05
"""
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    """
    1和2是用回归的思路来做二分类问题
    """
    mode = 2
    # load or create your dataset
    df_train = pd.read_csv('dataset/binary.train', header=None, sep='\t')
    df_test = pd.read_csv('dataset/binary.test', header=None, sep='\t')
    W_train = pd.read_csv('dataset/binary.train.weight', header=None)[0]
    W_test = pd.read_csv('dataset/binary.test.weight', header=None)[0]

    y_train = df_train[0]
    y_test = df_test[0]
    X_train = df_train.drop(0, axis=1)
    X_test = df_test.drop(0, axis=1)
    if mode == 3:
        """
        用GradientBoostingClassifier做二分类
        
        可以直接输出预测的分类值，也可以输出分类的概率
        """
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        prob = model.predict_proba(X_test)#Predict class probabilities for X.
        print(prediction[:5])#[1 0 0 1 0]
        print(prob[:5])
        """
[[0.27236011 0.72763989]
 [0.56744508 0.43255492]
 [0.72466762 0.27533238]
 [0.46670385 0.53329615]
 [0.78176948 0.21823052]]
 
        可以看到，第2列和直接用GradientBoostingRegressor预测的为1的概率是很接近的
        """
        print(model.feature_importances_)
        """
[0.02859148 0.01085152 0.00309006 0.03645152 0.00030362 0.08788641
 0.007849   0.00458998 0.00075369 0.02667845 0.00995686 0.0059722
 0.00062221 0.01333656 0.00536703 0.00608781 0.         0.01440449
 0.01060861 0.00472889 0.00041166 0.01455774 0.09022242 0.00596075
 0.08322082 0.29384467 0.11650031 0.11715124]
        """
    if mode == 2:
        """
        利用最优参数，用全部的train来训练模型；
        并在test上做测试
        """
        model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=3)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        print(prediction[:5])#[0.70109732 0.42378236 0.26383924 0.51261823 0.26064675]
        print(math.sqrt(mean_squared_error(y_test, prediction)))#0.4186670766971971
        print(model.feature_importances_)
        """
[0.02522196 0.0088327  0.00354536 0.03600168 0.00083261 0.08677325
 0.00654341 0.00342165 0.0016567  0.02288524 0.01033184 0.00626276
 0.00100435 0.01611042 0.00694322 0.00664558 0.         0.01640558
 0.0100925  0.00664581 0.         0.01519783 0.09153796 0.00822708
 0.07702901 0.29726059 0.11620089 0.11839002]
        """
    if mode == 1:
        """
        利用网格选择最优参数
        """
        estimator = GradientBoostingRegressor()

        param_grid = {
            'learning_rate': [0.01, 0.1, 1],
            'n_estimators': [80, 100, 120],
            'max_depth': [3, 5, 7]
        }
        gbm = GridSearchCV(estimator, param_grid, cv=3)
        gbm.fit(X_train, y_train)

        print('Best parameters found by grid search are:', gbm.best_params_)
        """
        Best parameters found by grid search are: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
        """
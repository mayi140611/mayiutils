#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: lightgbm_wrapper.py
@time: 2019-04-23 11:58

https://github.com/mayi140611/LightGBM
https://cloud.tencent.com/developer/news/375910
Light Gradient Boosting Machine (LightGBM)
是一个由微软亚洲研究院分布式机器学习工具包（DMTK）团队开源的基于决策树算法的分布式梯度提升（Gradient Boosting Decision Tree，GBDT）框架。

GBDT是机器学习中的一个非常流行并且有效的算法模型，它是一个基于决策树的梯度提升算法。

但是大训练样本和高维度特征的数据环境下，GBDT算法的性能以及准确性却面临了极大的挑战。

为了解决这些问题，LightGBM应运而生，不相上下的准确率，更快的训练速度，占用更小的内存

https://lightgbm.readthedocs.io/en/latest/Python-Intro.html
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import json
import pickle



if __name__ == '__main__':
    mode = 5
    print('Loading data...')
    if mode == 5:
        """
        https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py
        
        """
        # load or create your dataset
        df_train = pd.read_csv('dataset/binary.train', header=None, sep='\t')
        df_test = pd.read_csv('dataset/binary.test', header=None, sep='\t')
        W_train = pd.read_csv('dataset/binary.train.weight', header=None)[0]
        W_test = pd.read_csv('dataset/binary.test.weight', header=None)[0]

        y_train = df_train[0]
        y_test = df_test[0]
        X_train = df_train.drop(0, axis=1)
        X_test = df_test.drop(0, axis=1)

        num_train, num_feature = X_train.shape

        # create dataset for lightgbm
        # if you want to re-use data, remember to set free_raw_data=False
        lgb_train = lgb.Dataset(X_train, y_train,
                                weight=W_train, free_raw_data=False)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,
                               weight=W_test, free_raw_data=False)
        """
        weight 每个样本的权重
        """

        # specify your configurations as a dict
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }

        # generate feature names
        feature_name = ['feature_' + str(col) for col in range(num_feature)]

        print('Starting training...')
        # feature_name and categorical_feature
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=10,
                        valid_sets=lgb_train,  # eval training data
                        feature_name=feature_name,
                        categorical_feature=[21])

        print('Finished first 10 rounds...')
        # check feature name
        print('7th feature name is:', lgb_train.feature_name[6])#7th feature name is: feature_6

        print('Saving model...')
        # save model to file
        gbm.save_model('model.txt')
    # load or create your dataset
    df_train = pd.read_csv('dataset/regression.train', header=None, sep='\t')
    df_test = pd.read_csv('dataset/regression.test', header=None, sep='\t')

    y_train = df_train[0]
    y_test = df_test[0]
    X_train = df_train.drop(0, axis=1)
    X_test = df_test.drop(0, axis=1)
    if mode == 4:
        """
        https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/sklearn_example.py
Find best parameters for the model with sklearn's GridSearchCV
        
        """
        estimator = lgb.LGBMRegressor(num_leaves=31)

        param_grid = {
            'learning_rate': [0.01, 0.1, 1],
            'n_estimators': [20, 40]
        }

        gbm = GridSearchCV(estimator, param_grid, cv=3)
        gbm.fit(X_train, y_train)

        print('Best parameters found by grid search are:', gbm.best_params_)
        """
        Best parameters found by grid search are: {'learning_rate': 0.1, 'n_estimators': 40}
        """
    if mode == 3:
        """
        https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/sklearn_example.py
Create data for learning with sklearn interface
Basic train and predict with sklearn interface
Feature importances with sklearn interface
Self-defined eval metric with sklearn interface
        
        """
        print('Starting training...')
        # train
        gbm = lgb.LGBMRegressor(num_leaves=31,
                                learning_rate=0.05,
                                n_estimators=20)

        # self-defined eval metric
        # f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
        # Root Mean Squared Logarithmic Error (RMSLE)
        def rmsle(y_true, y_pred):
            return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False


        print('Starting training with custom eval function...')
        # train
        gbm.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric=rmsle,
                early_stopping_rounds=5)


        # another self-defined eval metric
        # f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
        # Relative Absolute Error (RAE)
        def rae(y_true, y_pred):
            return 'RAE', np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(np.mean(y_true) - y_true)), False


        print('Starting training with multiple custom eval functions...')
        # train
        gbm.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric=lambda y_true, y_pred: [rmsle(y_true, y_pred), rae(y_true, y_pred)],
                early_stopping_rounds=5)

        print('Starting predicting...')
        # predict
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
        # eval
        print('The rmsle of prediction is:', rmsle(y_test, y_pred)[1])
        print('The rae of prediction is:', rae(y_test, y_pred)[1])
        # feature importances
        print('Feature importances:', list(gbm.feature_importances_))
    if mode == 2:
        """
        https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/sklearn_example.py
Create data for learning with sklearn interface
Basic train and predict with sklearn interface
Feature importances with sklearn interface
        """
        print('Starting training...')
        # train
        gbm = lgb.LGBMRegressor(num_leaves=31,
                                learning_rate=0.05,
                                n_estimators=20)
        gbm.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='l1',
                early_stopping_rounds=5)

        print('Starting predicting...')
        # predict
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
        # eval
        print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)#0.4441153344254208

        # feature importances
        print('Feature importances:', list(gbm.feature_importances_))
        """
        [23, 7, 0, 33, 5, 56, 9, 1, 1, 21, 2, 5, 1, 19, 9, 6, 1, 10, 4, 10, 0, 31, 61, 4, 48, 102, 52, 79]
        """
    if mode == 1:
        """
        https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py
Construct Dataset
Basic train and predict
Eval during training
Early stopping
Save model to file
        0-1 二分类问题
        """

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        """
        reference : Dataset or None, optional (default=None)
            If this is Dataset for validation, training data should be used as reference.
        """

        # specify your configurations as a dict
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'l1'},
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }

        print('Starting training...')
        # train
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=20,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=5)

        print('Saving model...')
        # save model to file
        gbm.save_model('model.txt')

        print('Starting predicting...')
        # predict
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        print(y_pred[:10])#输出的是概率
        """
[0.6591194  0.52188659 0.38268875 0.5110434  0.38197134 0.34018169
 0.41396358 0.39943774 0.6229651  0.47306251] 
        """
        # eval
        print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)



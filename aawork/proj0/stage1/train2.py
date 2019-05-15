#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: train2.py
@time: 2019-05-15 18:22
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import classification_report


if __name__ == '__main__':
    df = pd.read_excel('train_0515.xlsx', index_col=0)
    # print(df.head())
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=14)
    # print(X_train.head())
    ### 数据转换
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_train, y_train, reference=lgb_train, free_raw_data=False)
    print('设置参数')
    params = {
        'boosting_type': 'gbdt',
        'boosting': 'dart',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.01,
        'num_leaves': 25,
        'max_depth': 3,
        'max_bin': 10,
        'min_data_in_leaf': 8,
        'feature_fraction': 0.6,
        'bagging_fraction': 1,
        'bagging_freq': 0,
        'lambda_l1': 0,
        'lambda_l2': 0,
        'min_split_gain': 0
    }
    print("开始训练")
    gbm = lgb.train(params,  # 参数字典
                    lgb_train,  # 训练集
                    num_boost_round=2000,  # 迭代次数
                    valid_sets=lgb_eval,  # 验证集
                    early_stopping_rounds=30)  # 早停系数
    preds_offline = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    p = pd.Series(preds_offline)
    p[p>=0.5] = 1
    p[p<0.5] = 0

    print(classification_report(y_test, p))


    preds_offline = gbm.predict(df.iloc[:, :-1], num_iteration=gbm.best_iteration)
    df['score'] = preds_offline

    df['s0.45'] = 1
    df.loc[df['score'] < 0.45, 's0.45'] = 0
    print(classification_report(df['label'], df['s0.45']))

    df['s0.5'] = 1
    df.loc[df['score'] < 0.5, 's0.5'] = 0
    print(classification_report(df['label'], df['s0.5']))

    df['s0.55'] = 1
    df.loc[df['score'] < 0.55, 's0.55'] = 0
    print(classification_report(df['label'], df['s0.55']))

    df.to_excel('df0515.xlsx', index=True)
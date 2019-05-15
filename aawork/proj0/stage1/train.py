#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: train.py
@time: 2019-04-19 11:52
"""
import pandas as pd
import numpy as np
from mayiutils.pickle_wrapper import PickleWrapper as picklew
from mayiutils.algorithm.algorithmset.calcPearson import calcPearson
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, f1_score
import graphviz
import xgboost
import os
import math
import itertools
# from feature_selector import FeatureSelector
import lightgbm as lgb
from sklearn import metrics


if __name__ == '__main__':
    mode = 9
    df = picklew.loadFromFile('train_data2.pkl')
    print(df.info())
    # print(df.head())
    X = df.values
    print(X.shape)
    y = picklew.loadFromFile('label.pkl')
    # print(y.value_counts())
    # y = y[:, np.newaxis]
    # print(list(y))
    y = np.array(list(y))

    if mode == 9:
        """
        结果评估
        """
        pred = pd.read_csv('tt.csv', header=None)
        # print(pred[:5])
        df = pd.DataFrame()
        df['score'] = pred.iloc[:, 1]
        df['s0.4'] = 1
        df.loc[df['score']<0.4, 's0.4']=0
        print(df['s0.4'].value_counts())
        print(classification_report(y, list(df['s0.4'])))

        df['s0.5'] = 1
        df.loc[df['score']<0.5, 's0.5']=0
        print(df['s0.5'].value_counts())
        print(classification_report(y, list(df['s0.5'])))
        """
0    421
1    141
Name: s0.5, dtype: int64
              precision    recall  f1-score   support

           0       0.98      0.95      0.96       432
           1       0.85      0.92      0.89       130
        """

        df['s0.6'] = 1
        df.loc[df['score']<0.6, 's0.6']=0
        print(df['s0.6'].value_counts())
        print(classification_report(y, list(df['s0.6'])))

        df['s0.7'] = 1
        df.loc[df['score']<0.7, 's0.7']=0
        print(df['s0.7'].value_counts())
        print(classification_report(y, list(df['s0.7'])))
    if mode == 8:
        """
        使用lightgbm, 输出概率
        """
        ### 数据转换
        lgb_train = lgb.Dataset(X, y, free_raw_data=False)
        lgb_eval = lgb.Dataset(X, y, reference=lgb_train, free_raw_data=False)
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
            num_boost_round = 2000,  # 迭代次数
            valid_sets = lgb_eval,  # 验证集
            early_stopping_rounds = 30)  # 早停系数
        preds_offline = gbm.predict(X, num_iteration=gbm.best_iteration)  # 输出概率
        print(preds_offline)
        pd.Series(preds_offline).to_csv('tt.csv')
    if mode == 7:
        """
        使用feature-selector得到特征重要性
        """
        fs = FeatureSelector(data=df, labels=y)
        fs.identify_collinear(correlation_threshold=0.975)
        correlated_features = fs.ops['collinear']
        print(correlated_features)
        # fs.plot_collinear()
        # fs.plot_collinear(plot_all=True)
        print(fs.record_collinear)
        # 4. Zero Importance Features
        fs.identify_zero_importance(task='classification', eval_metric='auc',
                                    n_iterations=10, early_stopping=True)
        one_hot_features = fs.one_hot_features
        base_features = fs.base_features
        print('There are %d original features' % len(base_features))
        print('There are %d one-hot features' % len(one_hot_features))
        zero_importance_features = fs.ops['zero_importance']
        print(zero_importance_features[:15])
        # fs.plot_feature_importances(threshold=0.99, plot_n=12)
        # print(fs.feature_importances)
        # fs.feature_importances.to_csv('fs_rs.csv', encoding='gbk')
        df_removed = fs.remove(methods=['collinear', 'zero_importance'])
        print(df_removed.shape)
        picklew.dump2File(df_removed, 'train_fs_removed.pkl')
    if mode == 6:
        """
        rf求特征重要性
        """
        rfmodel = RandomForestClassifier(n_estimators=80)
        rfmodel.fit(X, y)
        rs = pd.Series(rfmodel.feature_importances_, index=df.columns).sort_values(ascending=False)
        rs.to_csv('randomforest_rs.csv', encoding='gbk')
    if mode == 5:
        """
        计算皮尔逊相关系数
        """
        r = np.apply_along_axis(lambda x: calcPearson(x, y), axis=0, arr=X)
        print(r)
        rs = pd.Series(r, index=df.columns).sort_values(ascending=False)
        print(rs)
        rs.to_csv('pearson_rs.csv', encoding='gbk')
    if mode == 4:
        """
        whole xgboost train
        """
        model = xgboost.XGBClassifier(learning_rate=0.05, n_estimators=80, max_depth=7)
        model.fit(X, y)
        prediction = model.predict(X)
        # print(prediction)
        print(classification_report(y, prediction))
        # f1 = f1_score(y, prediction)
        print(model.feature_importances_)
        rs = pd.Series(model.feature_importances_, index=df.columns).sort_values(ascending=False)
        print(rs)
        rs.to_csv('xgboost_rs.csv', encoding='gbk')
    if mode == 3:
        """
        xgboost
        """
        skf = StratifiedKFold(n_splits=4)
        lr = [0.05, 0.1, 0.2]
        max_depth = [3, 5, 7]
        n_estimators = [80, 100, 120]
        # lr = [0.1, 0.12]
        # max_depth = [5, 6, 7]
        # n_estimators = [110, 120, 130]
        for l, n, m in itertools.product(lr, n_estimators, max_depth):
            print(l, n, m)
            f1 = []
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = xgboost.XGBClassifier(learning_rate=l, n_estimators=n, max_depth=m)
                model.fit(X_train, y_train)
                prediction = model.predict(X_test)
                # print(prediction)
                # print(classification_report(y_test, prediction))
                f1 = f1_score(y_test, prediction)
            print(np.mean(f1))

    if mode == 2:
        """
        dt whole train
        """
        clf = DecisionTreeClassifier(max_depth=4)
        # 拟合模型
        clf.fit(X, y)
        y_p = clf.predict(X)
        print(classification_report(y, y_p))
        print(clf.feature_importances_)

        rs = pd.Series(clf.feature_importances_, index=df.columns).sort_values(ascending=False)
        print(rs)
        rs.to_csv('dt_rs.csv', encoding='gbk')
        # dot_data = tree.export_graphviz(clf, out_file=None,
        #                                 feature_names=df.columns,
        #                                 # class_names=iris.target_names,
        #                                 filled=True, rounded=True,
        #                                 special_characters=True)
        # graph = graphviz.Source(dot_data)
        # graph.view()
    if mode == 1:
        """
        dt
        """
        skf = StratifiedKFold(n_splits=4)
        max_depths = [3, 6, 9]
        """
        0.7333333333333334
0.5925925925925926
0.5384615384615384
        """
        max_depths = [2, 3, 4, 5]
        """
0.6575342465753423
0.7333333333333334
0.7540983606557378
0.6181818181818182
        """
        for max_depth in max_depths:
            f1 = []
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # 训练模型，限制树的最大深度
                clf = DecisionTreeClassifier(max_depth=max_depth)
                # 拟合模型
                clf.fit(X_train, y_train)
                y_p = clf.predict(X_test)
                # print(classification_report(y_test, y_p))
                f1 = f1_score(y_test, y_p)
            print(np.mean(f1))

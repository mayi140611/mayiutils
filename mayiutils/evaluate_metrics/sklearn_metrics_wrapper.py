#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: sklearn_metrics_wrapper.py
@time: 2019/3/18 17:12


模型评估
https://blog.csdn.net/shine19930820/article/details/78335550
sklearn 评价指标
1、Classification 分类
‘accuracy’	metrics.accuracy_score
‘average_precision’	metrics.average_precision_score
‘f1’	metrics.f1_score	for binary targets
‘f1_micro’	metrics.f1_score	micro-averaged
‘f1_macro’	metrics.f1_score	macro-averaged
‘f1_weighted’	metrics.f1_score	weighted average
‘f1_samples’	metrics.f1_score	by multilabel sample
‘neg_log_loss’	metrics.log_loss	requires predict_proba support
‘precision’ etc.	metrics.precision_score	suffixes apply as with ‘f1’
‘recall’ etc.	metrics.recall_score	suffixes apply as with ‘f1’
‘roc_auc’	metrics.roc_auc_score
2、Clustering
‘adjusted_rand_score’	metrics.adjusted_rand_score
3、Regression
‘neg_mean_absolute_error’	metrics.mean_absolute_error
‘neg_mean_squared_error’	metrics.mean_squared_error
‘neg_median_absolute_error’	metrics.median_absolute_error
‘r2’	metrics.r2_score
"""
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score, precision_recall_curve\
                            , recall_score, confusion_matrix
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, cosine_distances, cosine_similarity, paired_cosine_distances


if __name__ == '__main__':
    mode = 3
    submode = 202
    if mode == 3:
        """
        距离计算
        """
        a = np.array([[1, 0, 1], [1, 1, 1]])
        b = np.array([[1, 1, 1],
                      [1, 0, 1],
                      [1, 0, 0]])
        print(cosine_similarity(a, a))
        """
[[1.         0.81649658]
 [0.81649658 1.        ]]
        """
        print(cosine_similarity(a, b))
        """
[[0.81649658 1.         0.70710678]
 [1.         0.81649658 0.57735027]]
        """
    if mode == 2:
        """
        分类评价指标
        """
        if submode == 202:
            """
            二分类
            """
            y_true = np.array([0, 0, 1, 1])
            y_scores = np.array([0.1, 0.4, 0.35, 0.8])
            print(roc_auc_score(y_true, y_scores))#0.75
            print(average_precision_score(y_true, y_scores))#0.8333333333333333
            # print(precision_recall_curve())
        if submode == 201:
            """
            多分类
            """
            y_test = np.array([1, 2, 3, 3, 2, 1, 1, 3, 2, 2])
            prediction = np.array([1, 3, 3, 2, 3, 1, 3, 2, 2, 2])
            print(accuracy_score(y_test, prediction))#0.4
            print(classification_report(y_test, prediction))
            """
                support表示真实数据中该类的个数
                 precision    recall  f1-score   support
    
              1       1.00      0.67      0.80         3
              2       0.50      0.50      0.50         4
              3       0.25      0.33      0.29         3
    
            avg / total       0.57      0.50      0.53        10
            """
    if mode == 1:
        """
        回归评价指标
        """
        if submode == 101:
            """
            R2 决定系数（拟合优度）
            模型越好：r2→1
            模型越差：r2→0
            一个模型的R2 值为0还不如直接用平均值来预测效果好；而一个R2值为1的模型则可以对目标变量进行完美的预测。
            从0至1之间的数值，则表示该模型中目标变量中有百分之多少能够用特征来解释。
            模型也可能出现负值的R2，这种情况下模型所做预测有时会比直接计算目标变量的平均值差很多。
            """
            y_true, y_pred = [3, -0.5, 2, 7], [2.5, 0.0, 2, 8]
            print(r2_score(y_true, y_pred))#0.9486081370449679
            #计算均方误差
            precision_recall_curve(mean_squared_error(y_true, y_pred))
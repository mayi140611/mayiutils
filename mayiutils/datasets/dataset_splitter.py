#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: dataset_splitter.py
@time: 2019/4/14 15:16
"""
"""
数据拆分的步骤：
1、先用train_test_split把数据拆分成训练集和测试集；
2、再将训练集进一步使用kfold，分为训练集和验证集

注：测试集始终是不变的，不参与训练！！！
"""

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import numpy as np


if __name__ == '__main__':
    mode = 2

    if mode == 2:
        """
        K折交叉验证
        K-Folds cross-validator
            Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
             Each fold is then used once as a validation while the k - 1 remaining folds form the training set.
        Stratified（分层的） K-Folds cross-validator
            Provides train/test indices to split data in train/test sets.
            This cross-validation object is a variation of KFold that returns stratified folds. 
            The folds are made by preserving the percentage of samples for each class.
        """
        X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        y = np.array([0, 0, 1, 1])
        skf = StratifiedKFold(n_splits=2)
        #Returns the number of splitting iterations in the cross-validator
        print(skf.get_n_splits(X, y))#2
        for train_index, test_index in skf.split(X, y):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

    if mode == 1:
        """
        将数据集划分为 训练集和测试集
        """
        X, y = np.arange(10).reshape((5, 2)), range(5)
        """
        test_size: 测试集所占的比例；
        默认在切分前会shuffle
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=14)
        train_test_split(y, shuffle=False)

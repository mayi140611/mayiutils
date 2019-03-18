#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: param_select.py
@time: 2019/3/18 17:22

KFold
K-Folds cross-validator
Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
Each fold is then used once as a validation while the k - 1 remaining folds form the training set.

Stratified K-Folds cross-validator
Provides train/test indices to split data in train/test sets.
This cross-validation object is a variation of KFold that returns stratified folds.
The folds are made by preserving the percentage of samples for each class.
"""
from sklearn.model_selection import StratifiedKFold, KFold


if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=14)
    skf.get_n_splits(X, y)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
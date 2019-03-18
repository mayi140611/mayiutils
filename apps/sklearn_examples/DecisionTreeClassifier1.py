#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: DecisionTreeClassifier1.py
@time: 2019/3/18 10:36

https://www.cnblogs.com/pinard/p/6056319.html
"""
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

import graphviz


if __name__ == '__main__':
    # 仍然使用自带的iris数据
    iris = datasets.load_iris()
    X = iris.data[:, [0, 2]]
    y = iris.target

    # 训练模型，限制树的最大深度4
    clf = DecisionTreeClassifier(max_depth=4)
    # 拟合模型
    clf.fit(X, y)

    # 画图
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # 网格
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    print(Z)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.show()

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names[:2],
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.view()


#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: seaborn_wrapper.py
@time: 2019-04-26 10:17
"""
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    mode = 2


    if mode == 1:
        """
        
        """
        iris = load_iris()
        dataset = np.concatenate((iris.data, iris.target[:, np.newaxis]), axis=1)
        iris_df = pd.DataFrame(dataset, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class_'])
        plt.figure()
        sns.pairplot(iris_df.dropna(), hue='class_', vars=list(iris_df.columns)[:-1])
        plt.show()
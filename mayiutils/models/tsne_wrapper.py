#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: tsne_wrapper.py
@time: 2019-05-20 19:18
TSNE是由SNE衍生出的一种算法，SNE最早出现在2002年，它改变了MDS和ISOMAP中基于距离不变的思想，将高维映射到低维的同时，
尽量保证相互之间的分布概率不变，SNE将高维和低维中的样本分布都看作高斯分布，
而Tsne将低维中的坐标当做T分布，这样做的好处是为了让距离大的簇之间距离拉大，从而解决了拥挤问题。
流形学习 (Manifold Learning)
python--sklearn,聚类结果可视化工具TSNE
　　TSNE提供了一种有效的降维方式，让我们对高于2维数据的聚类结果以二维的方式展示出来：
"""
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    iris = sns.load_dataset("iris")
    print(iris['species'].unique())
    print(iris.head())
    tsne=TSNE()
    tsne.fit_transform(iris.iloc[:, :4])  #进行数据降维,降成两维
    #a=tsne.fit_transform(data_zs) #a是一个array,a相当于下面的tsne_embedding_
    print(tsne.embedding_.shape)  # (150, 2)
    tsne=pd.DataFrame(tsne.embedding_, index=iris.index) #转换数据格式
    d = tsne[iris['species'] == 'setosa']
    plt.plot(d[0], d[1], 'r.')
    d = tsne[iris['species'] == 'versicolor']
    plt.plot(d[0], d[1], 'go')
    d = tsne[iris['species'] == 'virginica']
    plt.plot(d[0], d[1], 'b*')
    plt.show()
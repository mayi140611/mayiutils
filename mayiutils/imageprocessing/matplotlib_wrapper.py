#!/usr/bin/python
# encoding: utf-8
"""
Installation Issue with matplotlib Python
    **RuntimeError**: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends.
        Solution：
            I assume you have installed the pip matplotlib, there is a directory in you root called ~/.matplotlib.
            Create a file ~/.matplotlib/matplotlibrc there and add the following code: backend: TkAgg
几种图的类型和用途
    A bar plot is a plot that presents categorical data with
        rectangular bars with lengths proportional to the values that they
        represent. A bar plot shows comparisons among discrete categories. One
        axis of the plot shows the specific categories being compared, and the
        other axis represents a measured value.
    A box plot is a method for graphically depicting groups of numerical
        data through their quartiles.
        The box extends from the Q1 to Q3 quartile values of the data,
        with a line at the median (Q2). The whiskers extend from the edges
        of box to show the range of the data. The position of the whiskers
        is set by default to 1.5*IQR (IQR = Q3 - Q1) from the edges of the
        box. Outlier points are those past the end of the whiskers.
"""
# 大致的步骤
# 先新建一个画布
# plt.figure()
# plt.figure(figsize=(10,10))#制定绘图区域的大小
# 把画布分割成几个子图区域，同时制定当前绘制的子图
# plt.subplot(2,3,1) ：表示子图分割成2行3列，当前的绘图区域第一块，也可以不使用逗号
# plt.subplot(232) 表示第一行中间的区域
# #绘图
# x = [1,2,3,4]
# y = [5,4,3,2]
# plt.subplot(231)
# plt.plot(x, y)#线图
# plt.subplot(232)
# plt.bar(x, y)#条形图
# plt.subplot(233)
# plt.barh(x, y)#水平条形图
# plt.subplot(234)
# plt.bar(x, y)
# y1 = [7,8,5,3]
# plt.bar(x, y1, bottom=y, color = 'r')#叠加条形图
# plt.subplot(235)
# plt.boxplot(x)#箱形图
# plt.subplot(236)
# plt.suptitle('Categorical Plotting')#子图名称
# plt.scatter(x,y)#散点图
# plt.axis([0, 6, 0, 20])#[xmin, xmax, ymin, ymax] 指定x，y两个坐标轴的范围
# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title('Histogram of IQ')
# plt.grid(True)#显示出网格
# 显示绘制的图形
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MatplotlibWrapper(object):
    @classmethod
    def plot(selt, x, y, fmt=None, ls=None, lw=None):
        '''
        https://matplotlib.org/tutorials/introductory/pyplot.html
        线图
        @x, y : array-like or scalar
        @fmt : str, optional
            A format string, e.g.
            'ro' for red circles.
            'r-'：red 实线
            'r--': red 虚线
            'bs'： blue squares，蓝色的方框
            'g^'： 绿色的三角形
        marker: 点的形状
            .   o   v   ^   <   >   s(square)   p(pentagon五边形）   h(hexagon六边形)   +   x   D(diamond)  *   |
        markersize: 点的大小
        markeredgecolor: 点的边框色
        markerfacecolor: 点的填充色
        @linestyle or ls: 线条风格
            ['solid' | 'dashed', 'dashdot', 'dotted'
            ``'-'`` | ``'--'`` | ``'-.'`` | ``':'``
            | (offset, on-off-dash-seq) |  | ``'None'`` | ``' '`` | ``''``]
        @linewidth or lw: float value in points
        c: color
        label:
        '''
        return plt.plot(x, y, fmt, ls, lw)

    def scatter(self, x, y, c=None, marker=None):
        """
        散点图
        :param x: x坐标list，array_like, The data positions.
        :param y: y坐标list，array_like, The data positions.
        :param c: color
        :param marker: 散点的形状
        plt.scatter(diabetes_X_test, diabetes_y_test,  c='red', marker='*')
        :return:
        """
        return plt.scatter(x, y, c=c, marker=marker)
    @classmethod
    def hist(cls, x, bins, density, color, alpha=0.75):
        """

        :param x:
        :param bins:
        :param density:
        :param clolor:
        :param alpha:
        :return:
        """
        n, bins, patches = plt.hist(x, bins, density=density, color=color, alpha=alpha)
        return n, bins, patches


def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2, -y ** 2)


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    """

    :param low_dim_embs: 二维数组 (n, 2)
    :param labels:
    :param filename:
    :return:
    """
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    # 设置图片大小
    plt.figure(figsize=(15, 15))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

    plt.savefig(filename)


if __name__ == '__main__':
    mode = 0
    if mode == 3:
        """
        条形图
        """
        x = ['a', 'b', 'c']
        y = [1, 2, 3]
        s = pd.Series(y, index=x)
        plt.figure()
        # plt.bar(x, y)
        plt.bar(s.index, s.values)
        plt.show()
    if mode == 2:
        """
        等高线图
        https://www.cnblogs.com/huanggen/p/7533088.html
        """
        n = 256
        x = np.linspace(-3, 3, n)
        # print(x)
        y = np.linspace(-3, 3, n)
        X, Y = np.meshgrid(x, y)
        # print(X)
        """
        接下来我们进行颜色填充。使用函数plt.contourf把颜色加进去，位置参数分别为:X，Y，f(X,Y)。透明度0.75，
        并将f(X,Y)的值对应到color map的暖色组中寻找对应颜色。
        """
        plt.contourf(X, Y, f(X, Y), 8, alpha=0.75, cmap=plt.cm.hot)
        # 把等高线描出来
        C = plt.contour(X, Y, f(X, Y), 8, colors='black')
        # 加入Label,inline控制是否在Label画在线里面，字体大小为10.：
        plt.clabel(C, inline=True, fontsize=10)
        # 并将坐标轴隐藏
        plt.xticks(())
        plt.yticks(())
        plt.show()
    if mode == 1:
        """
        画频率直方图
        """
        mu, sigma = 100, 15
        x = mu + sigma * np.random.randn(10000)
        n, bins, patches = plt.hist(x, 50, density=1, color='g', alpha=0.75)
        print(n, bins, patches)
        print(n.shape, bins.shape)
        plt.xlabel('Smarts')
        plt.ylabel('Probability')
        plt.title('Histogram of IQ')
        plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        plt.axis([40, 160, 0, 0.03])
        plt.grid(True)
        plt.show()
    if mode == 0:
        """
        plot 折线图 观察随着x变化，y值的变化。要求x值从小到大排好序
        """
        plt.rcParams['font.sans-serif'] = ['STFangsong']
        plt.rcParams['axes.unicode_minus'] = False
        x = np.linspace(0, 1.1, 100)
        y = np.sin(x)
        plt.figure(figsize=(9, 9))
        plt.title('y=sin(x)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim((0, 1))  # x轴显示范围
        plt.ylim((0, 1))  # y轴显示范围
        plt.xticks([0, 0.5], rotation=45)  # x轴显示刻度
        plt.yticks([0, 0.2, 0.7, 1])
        plt.plot(x, y, ls=':', label='hello啊')  # label 图例显示内容
        plt.legend(loc='best')  # upper center
        plt.savefig('a.pdf')  # 注意要在show之前保存，否则为空白
        plt.savefig('a.png')
        plt.savefig('a.jpg')
        plt.show()
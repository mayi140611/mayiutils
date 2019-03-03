#!/usr/bin/python
# encoding: utf-8

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
        @linestyle or ls: ['solid' | 'dashed', 'dashdot', 'dotted' | (offset, on-off-dash-seq) | ``'-'`` | ``'--'`` | ``'-.'`` | ``':'`` | ``'None'`` | ``' '`` | ``''``]
        @linewidth or lw: float value in points 
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


if __name__ == '__main__':
    mu, sigma = 100, 15
    x = mu + sigma * np.random.randn(10000)
    n, bins, patches = MatplotlibWrapper.hist(x, 50, density=1, color='g', alpha=0.75)
    print(n, bins, patches)
    print(n.shape, bins.shape)
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.show()
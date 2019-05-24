#!/usr/bin/python
# encoding: utf-8
"""
http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
Series： One-dimensional ndarray with axis labels (including time series).

"""
import pandas as pd
import matplotlib.pyplot as plt


class SeriesWrapper(object):
    def __init__(self):
        pass
    @classmethod
    def items(self,series):
        '''
        Iterator over (column name, Series) pairs.
        for i,v in series1.items():
            ……
        '''
        return series.items()

    @classmethod
    def append(self,s1, to_append, ignore_index=False, verify_integrity=False):
        '''
        Concatenate two or more Series.
        注意这里不同于list.append, 相当于list.extend
        to_append : Series or list/tuple of Series
        ignore_index : boolean, default False
            If True, do not use the index labels.
        verify_integrity : boolean, default False
            If True, raise Exception on creating index with duplicates
        '''
        return s1.append(to_append, ignore_index, verify_integrity)
    
    @classmethod
    def nonzero(self,series):
        '''
        Return the *integer* indices of the elements that are non-zero
        >>> s = pd.Series([0, 3, 0, 4], index=['a', 'b', 'c', 'd'])
        # same return although index of s is different
        >>> s.nonzero()
        (array([1, 3]),)
        >>> s.iloc[s.nonzero()[0]]
        b    3
        d    4
        dtype: int64
        '''
        return series.nonzero()
    @classmethod
    def nonzero_item(self,series):
        '''
        Return series中的非零元素
        >>> s = pd.Series([0, 3, 0, 4], index=['a', 'b', 'c', 'd'])
        # same return although index of s is different
        >>> s.nonzero()
        (array([1, 3]),)
        >>> s.iloc[s.nonzero()[0]]
        b    3
        d    4
        dtype: int64
        '''
        return series.iloc[series.nonzero()[0]]

    @classmethod
    def plot(cls, s, kind, title=None, rot=45):
        """
        Series绘图
        s.plot(kind='line')`` is equivalent to ``s.plot.line()
        :param s:
        :param kind:
            line: 折线图 x: s.index; y: s.values
            bar: 柱状图
            barh: 水平柱状图
            box: 箱线图
            hist： 频率直方图
        :return:
        """
        s.plot(kind, title=title, rot=rot)
        plt.show()


if __name__ == '__main__':
    mode = 5
    # series creation
    s = pd.Series([1.1, 2, 3, None, 4, 5, 4, '2010-04-14'])
    s1 = pd.Series(range(5))
    # print(s)
    """
0           1.1
1             2
2             3
3          None
4             4
5             5
6             4
7    2010-04-14
dtype: object
    """
    # print(s.index)  # RangeIndex(start=0, stop=8, step=1)
    # print(s.index.values)  # [0 1 2 3 4 5 6 7]
    # print(s.values)  # [1.1 2 3 None 4 5 4 '2010-04-14']
    # print(s.to_numpy())  # 和.values等价
    # print(s.tolist())  # [1.1 2 3 None 4 5 4 '2010-04-14']
    # print(s.to_list())  # [1.1 2 3 None 4 5 4 '2010-04-14']
    # print(s.dtype)  # object
    # print(s.dtypes)  # object 和dtype相同
    # print(s.ndim)  # 1 数据的维度，Series都是1维
    # print(s.shape)  # (8,)
    # print(s.size)  # 8
    # print(s.count())  # 7  非None值的数量
    # print(s.T)  # 其自身
    # print(s.empty)  # False  如果s为空，返回True
    # print(s.name)  # None

    # s.fillna('na', inplace=True)
    # print(s.dtype)  # object
    # print(s.astype('category'))  # 强制类型转换 int32, int64, category
    # print(s.astype('category').dtype)  # category
    # print(s.apply('I am a {}'.format))  # apply 和map差别不大
    # print(s.map('I am a {}'.format, na_action='ignore'))
    # print(9 in s)#True 指 9 是否在s的index中
    # print(s.str.len())
    # print(s.map(str))
    # print(s.map(str).str.len())
    # print(s.astype(str).str.len())
    # print(s.astype(str).str.get(0))  # 如果字符串，就去第一个字符，如果列表就取第一个元素
    # print(s.str.capitalize())
    # print(s.nunique())  # 4 唯一值的数量
    # print(s.nunique(dropna=False))  # 5 唯一值的数量, 把None也算作一个值
    # """
    # sort
    # """
    # print(s.sort_index())
    # print(s.sort_values())
    # print(s.value_counts().sort_values(ascending=False))

    # print(s1.where(s1>1, 0))  # 找出s1中符合条件的，不符合条件的置0
    # print(s1.mask(s1>1, 0))  # 找出s1中符合条件的置0，感觉mask更符合人的一般习惯
    if mode == 2:
        """
        selection
        """
        print(s[s < 3])
        s[s < 3] = 0
        s[s >= 3] = 1
        print(s)
        # print(s1.nonzero())#(array([0, 1, 2, 3, 4]),)
        """
FutureWarning: Series.nonzero() is deprecated and will be removed in a future version.
Use Series.to_numpy().nonzero() instead
        """
        print(s.to_numpy().nonzero())#(array([0, 1, 2, 3, 4]),)
        #获取非零值
        print(s[s.to_numpy().nonzero()[0]])
        print(s.isnull())
        """
0    False
1     True
2    False
3     True
4    False
5    False
dtype: bool
        """
        print(s1.notnull())
        """
0     True
1    False
2     True
3    False
4     True
5     True
dtype: bool
        """
        # 获取非空值
        print(s1[s1.notnull()])
    if mode == 4:
        obj1 = pd.Series([1, 2])
        obj2 = pd.Series([3, 4, 5])
        print(obj1+obj2)#对应索引的值相加
        """
        0    4.0
        1    6.0
        2    NaN
        dtype: float64
        """
        print([1, 2]+[3, 4, 5])#[1, 2, 3, 4, 5]
        print(obj1.append(obj2))
        """
        0    1
        1    2
        0    3
        1    4
        2    5
        dtype: int64       
        """


    if mode == 2111:
        tag = pd.Series({'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4})
        print(tag)
        #注意赋值技巧
        tag[:] = range(5, len(tag)+5)
        print(tag)
        # 可以按照给定的索引列表给值，很有用！
        print(tag[['b', 'e']])
        print(type(tag['m']), tag['m'], tag['m'].reshape(-1, 1))
    if mode == 5:
        # SeriesWrapper.plot(pd.Series(range(5)), 'hist', 45)
        SeriesWrapper.plot(pd.Series(range(5)), 'line', 45)
        SeriesWrapper.plot(pd.Series(range(5)), 'bar', 45)
        SeriesWrapper.plot(pd.Series(range(5)), 'barh', 45)
        SeriesWrapper.plot(pd.Series(range(5)), 'box', 45)


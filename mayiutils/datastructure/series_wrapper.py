#!/usr/bin/python
# encoding: utf-8
import pandas as pd
import matplotlib.pyplot as plt


class series_wrapper(object):
    """
    http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
    """
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
    def append(self,to_append, ignore_index=False, verify_integrity=False):
        '''
        Concatenate two or more Series.
        注意这里不同于list.append, 相当于list.extend
        to_append : Series or list/tuple of Series
        ignore_index : boolean, default False
            If True, do not use the index labels.
        verify_integrity : boolean, default False
            If True, raise Exception on creating index with duplicates
        '''
        return pd.Series(data, index, dtype, name, copy, fastpath)
    
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


if __name__ == '__main__':
    mode = 2222
    # series creation
    s = pd.Series([1, 2, 3, 4, 5, 4, 2, 1, 2, 221])
    s1 = pd.Series([1, None, 3, None, 4, 0])
    print(s)
    print(9 in s)#True
    print(221 in s)
    if mode == 1:
        """
        view
        """
        print(s.sort_index())
        print(s.value_counts().sort_values(ascending=False))
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
        print(s1.to_numpy())#[ 1. nan  3. nan  4.  0.]
        print(s1.to_numpy().nonzero())#(array([0, 1, 2, 3, 4]),)
        #获取非零值
        print(s1[s1.to_numpy().nonzero()[0]])
        print(s1.isnull())
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


        if s.empty:#如果s1为空
            print('ppp')
        plt.figure()
        s.hist()
        plt.show()
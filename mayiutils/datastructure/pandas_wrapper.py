#!/usr/bin/python
# encoding: utf-8
"""
pandas中关于DataFrame行，列显示不完全（省略）的解决办法
https://blog.csdn.net/weekdawn/article/details/81389865
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

pandas读取Excel文件，以0开头的数据，出现数据缺失
    df6 = pd.read_excel('82200946506.xlsx', converters={'出险人客户号': str})
"""
import pandas as pd


class pandas_wrapper(object):
    @classmethod
    def series_inverse(self, series):
        '''
        把series的index和values互换
        '''
        return pd.Series(series.index,index=series.values)


if __name__ == '__main__':
    mode = 1
    if mode == 1:
        """
        分类变量 pd.Categorical
            Parameters
            ----------
            values : list-like
                The values of the categorical. If categories are given, values not in
                categories will be replaced with NaN.
            categories : Index-like (unique), optional
                The unique categories for this categorical. If not given, the
                categories are assumed to be the unique values of `values` (sorted, if
                possible, otherwise in the order in which they appear).
            ordered : boolean, (default False)
                Whether or not this categorical is treated as a ordered categorical.
                If True, the resulting categorical will be ordered.
                An ordered categorical respects, when sorted, the order of its
                `categories` attribute (which in turn is the `categories` argument, if
                provided).
        """
        c = pd.Categorical([1, 2, 3, 1, 2, 3])
        print(type(c))#<class 'pandas.core.arrays.categorical.Categorical'>
        print(c)
        """
    [1, 2, 3, 1, 2, 3]
    Categories (3, int64): [1, 2, 3]
        """
        print(pd.Categorical([1, 2, 3, 1, 2, 3], categories=[1, 2]))
        """
[1, 2, NaN, 1, 2, NaN]
Categories (2, int64): [1, 2]
        """
        print(pd.Categorical(['a','b','c','a','b','c'], ordered=True, categories=['c', 'b', 'a']))
        """
    [a, b, c, a, b, c]
    Categories (3, object): [c < b < a]
    """
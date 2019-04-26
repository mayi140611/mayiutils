#!/usr/bin/python
# encoding: utf-8
import pandas as pd

class pandas_wrapper(object):
    def __init__(self):
        pass
    @classmethod
    def read_csv(self,filepath_or_buffer, sep=',', delimiter=None, header='infer', names=None, index_col=None, usecols=None):
        '''
        Read CSV (comma-separated) file into DataFrame
        '''
        return pd.read_csv(filepath_or_buffer, sep)
    @classmethod
    def build_series(self,data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):
        '''
        Series:One-dimensional ndarray with axis labels (including time series).
        :data: 可以是list or dict
        '''
        return pd.Series(data, index, dtype, name, copy, fastpath)

    @classmethod
    def build_df_from_dict(self, data=None, index=None, columns=None, dtype=None, copy=False):
        '''
        data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
            'year': [2000, 2001, 2002, 2001, 2002],
            'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
        data = {'Nevada': {2001: 2.4, 2002: 2.9},
            'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
        '''
        return pd.DataFrame(data)
    @classmethod
    def build_df_from_list(self, data=None, index=None, columns=None, dtype=None, copy=False):
        '''
        >>>pandas.DataFrame([[1, 2],[3,4]])
        	0	1
        0	1	2
        1	3	4
        '''
        return pd.DataFrame(data)
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
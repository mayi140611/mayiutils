#!/usr/bin/python
# encoding: utf-8
import pandas as pd
import numpy as np
from feature_selector import FeatureSelector


class DataframeWrapper(object):
    """
    http://pandas.pydata.org/pandas-docs/stable/reference/frame.html

    """
    
    '''
    #####################################
    DataFrame索引数据
    #####################################
    '''

    @classmethod
    def get_not_null_df(self,df,cname):
        '''
        获取df中某列不为空的全部数据
        cname: string
        '''
        return df[pd.notnull(df[cname])]
    @classmethod
    def rename(df, index=None, columns=None, copy=True, inplace=False, level=None):
        '''
        修改df的index和columns的名称，默认会返回一个新的DF
        Alter axes labels.

        Function / dict values must be unique (1-to-1). Labels not contained in
        a dict / Series will be left as-is（被遗弃）. Extra labels listed don't throw an
        error.
        >>> df4.rename(index={0:'01'})#修改索引名
        >>> concept.rename(columns={'code':'symbol','c_name':'concept'})#修改列名
        Parameters
        ----------
        index, columns : dict-like or function, optional
            dict-like or functions transformations to apply to
            that axis' values. Use either ``mapper`` and ``axis`` to
            specify the axis to target with ``mapper``, or ``index`` and
            ``columns``.
        copy : boolean, default True
            Also copy underlying data
        inplace : boolean, default False
            Whether to return a new DataFrame. If True then value of copy is
            ignored.
        level : int or level name, default None
            In case of a MultiIndex, only rename labels in the specified
            level.

        Returns
        -------
        renamed : DataFrame

        '''
        return df.rename(mapper=None, index=index, columns=columns, axis=None, copy=copy, inplace=inplace, level=level)
        
    @classmethod
    def sort_by_column(self,df,cname,axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last'):
        '''
        按照df的某一列的值进行排序
        cname: string
        '''
        return df.sort_values(by=cname,axis=axis, ascending=ascending, inplace=inplace, kind=kind, na_position=na_position)
    @classmethod
    def get_unique_col_values(self, df,cname):
        '''
        返回df的cname列的唯一值
        cname: string
        return: The unique values returned as a NumPy array. In case of categorical
        data type, returned as a Categorical.
        '''
        return df[cname].unique()
    
    @classmethod
    def itertuples(self, df):
        '''
        获取df的迭代器，按行迭代，每行是一个<class 'pandas.core.frame.Pandas'>对象，可以当做元组看待
        如：Pandas(Index=13286, user_id=5, item_id=385, rating=4, timestamp=875636185)
        for line in df.itertuples():
            #由于user_id和item_id都是从1开始编号的，所有要减一
            train_data_matrix[line[1], line[2]] = line[3] 
        line[0]是df的索引
        '''
        return df.itertuples()
    '''
    #####################################
    DataFrame合并
    #####################################
    '''
    def merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
        '''
        按照两个df相同的列名合并DataFrame
        Merge DataFrame objects by performing a database-style join operation by
        columns or indexes.
        Parameters
        ----------
        left : DataFrame
        right : DataFrame
        how : {'left', 'right', 'outer', 'inner'}, default 'inner'
            * left: use only keys from left frame, similar to a SQL left outer join;
              preserve key order
            * right: use only keys from right frame, similar to a SQL right outer join;
              preserve key order
            * outer: use union of keys from both frames, similar to a SQL full outer
              join; sort keys lexicographically
            * inner: use intersection of keys from both frames, similar to a SQL inner
              join; preserve the order of the left keys
        on : label or list
            根据两个df共有的列名进行合并
            Column or index level names to join on. These must be found in both
            DataFrames. If `on` is None and not merging on indexes then this defaults
            to the intersection of the columns in both DataFrames.
        left_on : label or list, or array-like
            如果根据两个df的不同的列名合并，则需要制定left_on&right_on
            Column or index level names to join on in the left DataFrame. Can also
            be an array or list of arrays of the length of the left DataFrame.
            These arrays are treated as if they are columns.
        right_on : label or list, or array-like
            Column or index level names to join on in the right DataFrame. Can also
            be an array or list of arrays of the length of the right DataFrame.
            These arrays are treated as if they are columns.
        left_index : boolean, default False
            也可以根据index进行合并
            Use the index from the left DataFrame as the join key(s). If it is a
            MultiIndex, the number of keys in the other DataFrame (either the index
            or a number of columns) must match the number of levels
        right_index : boolean, default False
            Use the index from the right DataFrame as the join key. Same caveats as
            left_index     
        sort : boolean, default False
            Sort the join keys lexicographically in the result DataFrame. If False,
            the order of the join keys depends on the join type (how keyword)
        suffixes : 2-length sequence (tuple, list, ...)
            如果合并的两个的df有相同的列名，则加后缀区分
            Suffix to apply to overlapping column names in the left and right
            side, respectively
        copy : boolean, default True
            If False, do not copy data unnecessarily
        indicator : boolean or string, default False
            If True, adds a column to output DataFrame called "_merge" with
            information on the source of each row.
            If string, column with information on source of each row will be added to
            output DataFrame, and column will be named value of string.
            Information column is Categorical-type and takes on a value of "left_only"
            for observations whose merge key only appears in 'left' DataFrame,
            "right_only" for observations whose merge key only appears in 'right'
            DataFrame, and "both" if the observation's merge key is found in both.

        validate : string, default None
            If specified, checks if merge is of specified type.

            * "one_to_one" or "1:1": check if merge keys are unique in both
              left and right datasets.
            * "one_to_many" or "1:m": check if merge keys are unique in left
              dataset.
            * "many_to_one" or "m:1": check if merge keys are unique in right
              dataset.
            * "many_to_many" or "m:m": allowed, but does not result in checks.
        '''
        return pd.merge(left, right, how=how, on=on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index, sort=sort, suffixes=suffixes, copy=copy, indicator=indicator, validate=validate)

    def join(left, other, on=None, how='left', lsuffix='', rsuffix='', sort=False):
        '''
        和merge函数类似，只不过调用的主体是left_df
        '''
        return left.join(other, on=None, how='left', lsuffix='', rsuffix='', sort=False)

    def concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=None, copy=True):
        '''
        如果axis=0，按照相同的column名进行列方向的叠加
        如果axis=1，按照相同的index名进行行方向的叠加
        '''
        return None

    @classmethod
    def isIn(cls, df, field, rowlist):
        """
        获取满足条件的样本。场景就是把df中某个字段满足要求的一些行取出来
        :param df:
        :param field:
        :param rowlist:
        :return:
        """
        return df.loc[df.loc[:, field].isin(rowlist)]


if __name__ == '__main__':
    mode = 0
    # 删除全部为null的列，如果只要出现null就删掉，则how=any
    # df2 = df1.dropna(axis=1, how='all')
    if mode == 5:
        data = {'state': [1, 1, 2, 2, 1, 2, 2], 'pop': ['a', 'b', 'c', 'd', 'b', 'c', 'd']}
        frame = pd.DataFrame(data)
        print(frame)
        """
        keep : {'first', 'last', False}, default 'first'
            - ``first`` : Drop duplicates except for the first occurrence.
            - ``last`` : Drop duplicates except for the last occurrence.
            - False : Drop all duplicates.
        """
        #保留某一特征中重复值的第一条
        a = frame.drop_duplicates(subset=['pop'], keep='first')
        print(a)
    if mode == 4:
        d = pd.DataFrame([[0, 0], [0, 1], [1, 1]])
        print(d == 1)#判断d中每一个元素是否为1
        """
               0      1
        0  False  False
        1  False   True
        2   True   True
        """
        d = (d == 1)
        print(len(d))#3 等价于df.shape[0]
        print(d.sum())# 按列求和，等价于d.sum(axis=0)
    if mode == 3:
        """
        DataFrame生成 
        可以通过Series, arrays, constants, or list-like objects生成df
        """
        # list
        df = pd.DataFrame([[4, 9], ] * 3, columns=['A', 'B'])
        # array
        df = pd.DataFrame(np.array([[4, 9], ] * 3), columns=['A', 'B'])
        print(df)
        """
           A  B
        0  4  9
        1  4  9
        2  4  9
        """
        # dict
        data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                'year': [2000, 2001, 2002, 2001, 2002],
                'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
        df = pd.DataFrame(data)
        print(df)
        # series
        s1 = pd.Series(5, index=['a', 'b', 'c', 'd', 'e'])
        print(pd.DataFrame(s1))
        """
           0
        a  5
        b  5
        c  5
        d  5
        e  5
        """
        s2 = pd.Series(range(6), index=['b', 'c', 'd', 'e', 'f', 'g'])
        print(pd.DataFrame([s1, s2]))
        """
             a    b    c    d    e    f    g
        0  5.0  5.0  5.0  5.0  5.0  NaN  NaN
        1  NaN  0.0  1.0  2.0  3.0  4.0  5.0
        """
        print(pd.DataFrame([s1, s2]).fillna(''))
        """
           a    b    c    d    e  f  g
        0  5  5.0  5.0  5.0  5.0      
        1     0.0  1.0  2.0  3.0  4  5
        """
        # data 为 None
        print(pd.DataFrame(index=['support', 'confidence']))
    if mode == 2:
        """
        apply
        Apply a function along an axis of the DataFrame.
        如果 axis=None，则应用函数在每个元素上，等同于applymap
        axis = 0，给每列的元素应用函数
        axis = 1，给每行的元素应用函数
        applymap函数：
        让函数作用在dataframe的每一个元素上
        """
        df = pd.DataFrame([[4, 9], ] * 3, columns=['A', 'B'])
        print(df)
        """
           A  B
        0  4  9
        1  4  9
        2  4  9
        """
        print(df.apply(np.sqrt))
        """
             A    B
        0  2.0  3.0
        1  2.0  3.0
        2  2.0  3.0
        """
        print(df.apply(np.sum, axis=0))
        """
        A    12
        B    27       
        """
        print(df.applymap(np.sqrt))
    if mode == 1:
        df =  pd.read_excel("疾病列表190126.xlsx")
        #获取满足要求的样本
        df1 = DataframeWrapper.isIn(df, '疾病名称（39）', ['嗜睡症', '脑萎缩'])
        print(df1)






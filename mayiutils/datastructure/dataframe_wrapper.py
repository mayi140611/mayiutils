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
    """
    http://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html
    """
    mode = 3
    """
    DF Creation
    """
    d = pd.DataFrame([[0, 0], [0, 1], [1, 1]])
    # 创建时间序列索引
    # dates = pd.date_range('20130101', periods=6)
    # print(dates)
    """
DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
               '2013-01-05', '2013-01-06'],
              dtype='datetime64[ns]', freq='D')
    """
    """
    load data
    """
    # load带有日期列的数据
    loans = pd.read_csv('../../tmp/loans.csv', parse_dates=['loan_start', 'loan_end'])
    # print(loans.head())
    """
   client_id loan_type  loan_amount  ...  loan_start   loan_end  rate
0      46109      home        13672  ...  2002-04-16 2003-12-20  2.15
1      46109    credit         9794  ...  2003-10-21 2005-07-17  1.25
2      46109      home        12734  ...  2006-02-01 2007-07-05  0.68
3      46109      cash        12518  ...  2010-12-08 2013-05-05  1.24
4      46109    credit        14049  ...  2010-07-07 2012-05-21  3.13
    """
    d1 = loans['loan_end'].dt
    # print(d1)#<pandas.core.indexes.accessors.DatetimeProperties object at 0x11639b518>
    # 获取month or year
    m = d1.month
    # print(type(m))#<class 'pandas.core.series.Series'>

    if mode == 1:
        """
        查看数据 
        Viewing Data
        
        head(),  默认前5个样本
        sample(): a random sample of items from an axis of object
        tail()：默认后5个sample
        
        info()
        describe()#shows a quick statistic summary of your data。注意，df会自动选择数值列进行统计
        
        df.T 转置
        """
        # loans.info()
        """
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 443 entries, 0 to 442
Data columns (total 8 columns):
client_id      443 non-null int64
loan_type      443 non-null object
loan_amount    443 non-null int64
repaid         443 non-null int64
loan_id        443 non-null int64
loan_start     443 non-null datetime64[ns]
loan_end       443 non-null datetime64[ns]
rate           443 non-null float64
dtypes: datetime64[ns](2), float64(1), int64(4), object(1)
memory usage: 27.8+ KB
        """
        # print(loans.describe())
        """
          client_id   loan_amount      repaid       loan_id        rate
count    443.000000    443.000000  443.000000    443.000000  443.000000
mean   38911.060948   7982.311512    0.534989  11017.101580    3.217156
std     7768.681063   4172.891992    0.499338    581.826222    2.397168
min    25707.000000    559.000000    0.000000  10009.000000    0.010000
25%    32885.000000   4232.500000    0.000000  10507.500000    1.220000
50%    39505.000000   8320.000000    1.000000  11033.000000    2.780000
75%    46109.000000  11739.000000    1.000000  11526.000000    4.750000
max    49624.000000  14971.000000    1.000000  11991.000000   12.620000
        
        Generate descriptive statistics that summarize the central tendency,
        dispersion and shape of a dataset's distribution, excluding
        ``NaN`` values.

        Analyzes both numeric and object series, as well
        as ``DataFrame`` column sets of mixed data types. The output
        will vary depending on what is provided. Refer to the notes
        below for more detail.
        """
        # print(loans.T)
        """
                             0    ...                  442
client_id                  46109  ...                26945
loan_type                   home  ...                 home
loan_amount                13672  ...                 3643
repaid                         0  ...                    0
loan_id                    10243  ...                11434
loan_start   2002-04-16 00:00:00  ...  2010-03-24 00:00:00
loan_end     2003-12-20 00:00:00  ...  2011-12-22 00:00:00
rate                        2.15  ...                 0.13
        """
        # print(loans.sort_index(ascending=False))#按照索引降序查看
        print(loans.sort_values(by='rate'))#按照rate列升序查看
        """
        修改column name
        整体修改
            df.columns = ['name1', 'name2'...]
        修改几个：
            df.rename
        """
    if mode == 2:
        """
        行列操作
        # 删除全部为null的列
        df2 = df1.dropna(axis=1, how='all')
        #删除某列
        del frame2['eastern']
        """
        pass
    if mode == 3:
        """
        合并
        merge
        concat
        """
        # 以clients的'client_id'和stats的索引进行left_join
        # clients.merge(stats, left_on = 'client_id', right_index=True, how = 'left')
        print(d)
        print(pd.concat([d, d], axis=0))#列方向合并
    if mode == 4:
        """
        groupby
        followed by a suitable aggregation function
            可以通过agg函数，但是不太灵活
            mean()
            max()
            min()
            quantile(0.9)#求每个分组的0.9分位数
        """
        # 同时使用多个聚合函数
        stats = loans.groupby(by='client_id')['loan_amount'].agg(['mean', 'max', 'min'])
        t = loans.groupby('client_id')
        # print(t)#<pandas.core.groupby.generic.DataFrameGroupBy object at 0x1a200fbbe0>
        stats.columns = ['mean_loan_amount', 'max_loan_amount', 'min_loan_amount']
        # print(type(stats))#<class 'pandas.core.frame.DataFrame'>
        # print(stats.head())
        """
           mean_loan_amount  max_loan_amount  min_loan_amount
client_id                                                    
25707           7963.950000            13913             1212
26326           7270.062500            13464             1164
26695           7824.722222            14865             2389
26945           7125.933333            14593              653
29841           9813.000000            14837             2778
        """
        tt = t['loan_amount'].quantile(0.9)
        # print(type(tt))#<class 'pandas.core.series.Series'>

        # 使用自定义函数进行聚合
        def peak_to_peak(arr):
            return arr.max() - arr.min()
        # print(t['loan_amount'].agg(peak_to_peak))
        # 查看每个分组的统计摘要信息
        # print(t['loan_amount'].describe())
        """
           count          mean          std  ...      50%       75%      max
client_id                                    ...                            
25707       20.0   7963.950000  4149.486062  ...   9086.0  11419.00  13913.0
26326       16.0   7270.062500  4393.666631  ...   6133.0  11270.75  13464.0
26695       18.0   7824.722222  4196.462499  ...   9084.5  10289.00  14865.0
26945       15.0   7125.933333  4543.621769  ...   8337.0   9289.00  14593.0

        """

    if mode == 5:
        """
        删除重复数据
        """
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
    if mode == 6:
        """
        求相关系数
        df.corr()  # pearson相关系数
        df.corr('kendall') # Kendall Tau相关系数
        df.corr('spearman') # spearman秩相关
        热力图表示
        sns.heatmap(df.corr())
        plt.show()
        """
        print(loans.corr())
        """
             client_id  loan_amount    repaid   loan_id      rate
client_id     1.000000     0.046507  0.085547 -0.025350  0.058672
loan_amount   0.046507     1.000000  0.012506  0.074782 -0.033340
repaid        0.085547     0.012506  1.000000 -0.076472 -0.016172
loan_id      -0.025350     0.074782 -0.076472  1.000000  0.010918
rate          0.058672    -0.033340 -0.016172  0.010918  1.000000
        """
        # 热力图表示

    if mode == 4444:

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
    if mode == 3333:
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
    if mode == 2222:
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
    if mode == 1111:
        df =  pd.read_excel("疾病列表190126.xlsx")
        #获取满足要求的样本
        df1 = DataframeWrapper.isIn(df, '疾病名称（39）', ['嗜睡症', '脑萎缩'])
        print(df1)






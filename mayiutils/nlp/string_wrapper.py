#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: string_wrapper.py
@time: 2019/1/27 20:40
"""


class StringWrapper:
    """
    主要是对python中的string的相关操作的封装
    """
    @classmethod
    def formatFloat(cls, strformat, inputs):
        """
        格式化浮点数
        .3：表示保留几位数字，如1.2345  .3  后是1.23
        .4f：表示保留几位小数。
        '{0:.3}__{1:.4f}'.format(1.23456,16.77777777777)
        结果为：'1.23__16.7778'
        :param strformat:
        :param inputlist:
        :return:
        """
        return strformat.format(inputs)


if __name__ == '__main__':
    mode = 2
    if mode == 2:
        """
        format
        """
        print('%-16s-%16s' %('123','456'))#123             -             456
        print('%-16s-%16s\n%-16s-%16s' % ('123456789', '123456789', '123456789', '123456789'))
        """
        123456789       -       123456789
        123456789       -       123456789       
        """
    if mode == 1:
        """
        切片 slice  reverse
        """
        s = '我是中国人'
        print(s[-1:1])
        print(s[:2])
        print( 0 or 2)
        #把一个int型数值按位展开，转换为整数数组
        n = 8657
        print(list(map(int, str(n))))#[8, 6, 5, 7]
        print('我是中国人'[::-1])  # 人国中是我




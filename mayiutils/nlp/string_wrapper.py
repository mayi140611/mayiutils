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









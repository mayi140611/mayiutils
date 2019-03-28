#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: basic_op.py
@time: 2019/3/7 9:55

python 基础函数操作
"""
import math
import numpy as np


class BasicOpWrapper:
    @classmethod
    def round(cls, number, ndigits=0):
        """
        把number四舍五入为指定的小数位数
        :param number:
        :param ndigits:
        :return:
        """
        return round(number, ndigits)


if __name__ == '__main__':
    mode = 1

    if mode == 3:
        """
        map func
        """
        arr = np.arange(8).reshape((2, 4))
        print(arr)
        """
        [[0 1 2 3]
        [4 5 6 7]]
        """
        b = map(lambda x: sum(x), arr)#按照第一个维度（行）展开，每一行是一个元素
        print(list(b))
        """
        [6, 22]
        """
    if mode == 2:
        """
        取整
        """
        a = 1.4
        b = 1.5
        print(round(a), round(b))#1 2 四舍五入
        print(int(a), int(b))#向下取整
        print(math.floor(a), math.floor(b))#向下取整
        print(math.ceil(a), math.ceil(b))#向上取整
    if mode == 1:
        """
        强制类型转换
        """
        print(5 % 2)
        print(2 ** 31-1)
        print(int('-5'))#-5
        print(bool('True'))#True
        print(bool('False'))#True
        print(bool('true'))#True
        print(bool('false'))#True










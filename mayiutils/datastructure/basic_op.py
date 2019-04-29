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
    mode = 5
    if mode == 5:
        """
        复杂for循环
        """
        l1 = np.arange(48).reshape(12, 4)
        print(l1.tolist())#ndarray转list
        #flatten
        l2 = [j for i in l1 for j in i]
        print(l2)
    if mode == 4:
        """
        and or
        and 与or返回的不是bool型,而是原值
        2.and 为假时，返回第一个为假的值，（因为只要检测一个为假就能确定返回结果了）

        3.and为真时，返回最后一个为真的值，（因为只有检测到最后一个为True时才能确定返回结果）
        
        4.or为真时，返回第一个为真的值，（因为只要一个为真就可以确定返回结果了，直接返回检测到的值）
        
        5.or为假时，返回最后一个为假的值，（因为必须检测没有一个真值，才会确定返回结果）
        
        个人认为使用了成本最低理论，即返回确定最终结果的值
        """
        print(0 or 2)
        print(1 or 2)
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










#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: basic_type.py
@time: 2019/3/7 9:55

python 基础类型
"""
import math




#类型转换Type conversion
if __name__ == '__main__':
    mode = 2
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
        print(bool('True'))#True
        print(bool('False'))#True
        print(bool('true'))#True
        print(bool('false'))#True










#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: multiprocessing_wrapper.py
@time: 2019/3/14 21:34
"""
import multiprocessing


if __name__ == '__main__':
    #获取CPU核数
    print(multiprocessing.cpu_count())#4
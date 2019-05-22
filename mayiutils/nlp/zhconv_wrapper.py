#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: zhconv_wrapper.py
@time: 2019-05-21 18:19

中文简繁转换
"""
from zhconv import convert


if __name__ == '__main__':
    print(convert('歐幾里得西元前三世紀的古希臘數學家。', 'zh-cn'))  # 我干什么不干你事。
    print(convert('人体内存在很多微生物', 'zh-tw'))  # 人體內存在很多微生物
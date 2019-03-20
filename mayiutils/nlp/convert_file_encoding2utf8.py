#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: convert_file_encoding2utf8.py
@time: 2019/3/19 21:09

批量将文件编码改为UTF8的python小程序
https://blog.csdn.net/lee0_king/article/details/81782317
"""

import os
from chardet import detect


if __name__ == "__main__":
    fns = []
    file_name = os.listdir('D:/Desktop/SogouCS.reduced')
    for fn in file_name:
        if fn.endswith('.txt'):  # 这里填文件后缀
            fns.append(os.path.join('D:/Desktop/SogouCS.reduced', fn))

    for fn in fns:
        with open(fn, 'rb+') as fp:
            content = fp.read()
            codeType = detect(content)['encoding']
            print(codeType)
            content = content.decode(codeType, "ignore").encode("utf8")
            fp.seek(0)
            fp.write(content)
            print(fn, "：已修改为utf8编码")

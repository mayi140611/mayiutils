#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: fileoperation_wrapper.py
@time: 2019/2/21 10:14
"""
class FileOperationWrapper:
    @classmethod
    def writeList2File(cls, list, filepath, encoding='utf8'):
        """
        把list写入文件
        :param list:
        :param filepath:
        :param encoding:
        :return:
        """
        with open(filepath, 'w+', encoding=encoding) as f:
            f.writelines(list)



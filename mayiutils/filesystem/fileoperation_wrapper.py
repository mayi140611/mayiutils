#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: fileoperation_wrapper.py
@time: 2019/2/21 10:14

参考链接：https://docs.python.org/3.6/library/io.html
"""


class FileOperationWrapper:
    """
    文件访问模式：
        r：以只读方式打开文件，文件指针放在开头
        rb：以二进制格式打开一个文件用于只读
        r+: 打开一个文件用于读写
        rb+
        w: 打开一个文件只用于写入，如果文件存在，则覆盖，如果不存在，创建新文件
        wb
        w+
        wb+
        a: 打开一个文件用于追加，文件指针放在末尾。如果文件不存在，创建新文件
        ab
        a+
        ab+
    """
    @classmethod
    def read(cls, f, size=-1):
        """
        Read up to size bytes from the object and return them.
        As a convenience, if size is unspecified or -1, all bytes until EOF are returned.
        :param f:
        :return:
        """
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

if __name__ == '__main__':
    mode = 3
    if mode == 3:
        # 打开一个文件
        fo = open("foo.txt", "r+")
        print(type(fo))#<class '_io.TextIOWrapper'>
        str = fo.read(10);
        print("读取的字符串是 : ", str)
        # 关闭打开的文件
        fo.close()
    if mode == 2:
        fo = open("foo.txt", "a")
        print("文件名: ", fo.name)#foo.txt
        print("是否已关闭 : ", fo.closed)#False
        print("访问模式 : ", fo.mode)#a
        fo.write("1www.runoob.com!\nVery good site!\n");
        fo.close()
    if mode == 1:
        # 键盘输入
        str = input("Please enter:");
        print("你输入的内容是: ", str)


#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: os_wrapper.py
@time: 2019/2/22 14:21
#主要是对python中的os的相关操作的封装
"""
import os


class OsWrapper(object):
    def __init__(self):
        pass

    @classmethod
    def abspath(self, path):
        """
        获取绝对路径
        >>>os.path.abspath('.')
        '/home/ian/code/github/utils/0examples'
        :param path:
        :return:
        """
        return os.path.abspath(path)

    @classmethod
    def dirname(self, path):
        '''
        获取路径所在的文件夹
        >>>os.path.dirname('ian/code/github/utils/0examples/')
        'ian/code/github/utils/0examples'
        >>>os.path.dirname('ian/code/github/utils/0examples')
        'ian/code/github/utils'
        '''
        return os.path.dirname(path)

    @classmethod
    def join(self, path, *paths):
        '''
        Join two or more pathname components, inserting '/' as needed.
        If any component is an absolute path, all previous path components
        will be discarded.  An empty last part will result in a path that
        ends with a separator.
        >>>os.path.join('ian/code/','github/utils/0examples')
        'ian/code/github/utils/0examples'
        >>>os.path.join('ian/code','github/utils/0examples')
        'ian/code/github/utils/0examples'
        >>>os.path.join('ian/code','/github/utils/0examples')
        '/github/utils/0examples'
        '''
        return os.path.join(path, *paths)

    @classmethod
    def listDir(cls, filepath):
        """
        列出filepath下所有文件名和文件夹名，不递归
        Return a list containing the names of the entries in the directory given by path.
        :param filepath:
        :return:
        """
        return os.listdir(filepath)

    @classmethod
    def rename(cls, src, dst):
        """
        重命名文件或文件夹
        :param src:
        :param dst:
        :return:
        """
        return os.rename(src, dst)

    @classmethod
    def mkdir(cls, filepath, mode):
        """

        :param filepath:
        :param mode:
        :return:
        """
        return os.mkdir(filepath, mode)

    @classmethod
    def remove(cls, filepath):
        """
        Remove (delete) the file path.
        If path is a directory, OSError is raised. Use rmdir() to remove directories.
        :param filepath:
        :return:
        """
        return os.remove(filepath)

    @classmethod
    def rmdir(cls, filepath):
        """
        Remove (delete) the file path.
        If path is a directory, OSError is raised. Use rmdir() to remove directories.
        :param filepath:
        :return:
        """
        return os.rmdir(filepath)
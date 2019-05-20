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
import shutil


class OsWrapper(object):
    def __init__(self):
        pass

    @classmethod
    def abspath(self, path):
        """
        输入相对路径，获取绝对路径
        os.path.abspath('.')
        :param path: 相对路径
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
    def walk(cls, top, topdown=True, onerror=None, followlinks=False):
        """
        遍历文件夹， 返回一个generator对象
        Generate the file names in a directory tree by walking the tree either top-down or bottom-up.
        For each directory in the tree rooted at directory top (including top itself),
        it yields a 3-tuple (dirpath, dirnames, filenames).
        :param topdown:
        :param onerror:
        :param followlinks:
        :return:
        """
        return os.walk(top, topdown, onerror, followlinks)

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
    def mkdir(cls, filepath):
        """
        创建目录
        os.mkdir('reslut/log')# FileNotFoundError: [WinError 3] 系统找不到指定的路径。: 'reslut/log'
        :param filepath:
        :param mode:
        :return:
        """
        return os.mkdir(filepath)

    @classmethod
    def mkdirs(cls, filepath):
        """
        Super-mkdir; create a leaf directory and all intermediate ones.
        os.makedirs('reslut/log')
        :param filepath:
        :param mode:
        :return:
        """
        return os.makedirs(filepath)

    @classmethod
    def mv(cls, src, dst):
        """
        Recursively move a file or directory to another location. This is
        similar to the Unix "mv" command.
        shutil.move('a.txt', 'result/log/a10.txt')
        :param src:
        :param dst:
        :return: Return the file or directory's destination.
        """
        return shutil.move(src, dst)
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

    @classmethod
    def exists(cls, filepath):
        """
        os.path.exists('../file_io')
        :param filepath:
        :return:
        """
        return os.path.exists(filepath)

if __name__=='__main__':
    # 获取文件相关属性
    statinfo = os.stat('os_wrapper.py')
    print(statinfo)
    """
    os.stat_result(st_mode=33188, st_ino=12462088, st_dev=16777220, st_nlink=1, st_uid=502, 
    st_gid=20, st_size=4673, st_atime=1558342084, st_mtime=1558342084, st_ctime=1558342084)
    """
    # 获取文件大小
    print(statinfo.st_size)  # 4673

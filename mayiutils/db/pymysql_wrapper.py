#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: pymysql_wrapper.py
@time: 2019/1/30 18:36
PyMySQL==0.9.2
pip install pymysql
"""
import pymysql

class PyMysqlWrapper:
    def __init__(self, user='root',passwd='!Aa123456',db='daozhen',host='127.0.0.1',use_unicode=True,charset='utf8'):
        """
        创建连接、创建游标
        :param user:
        :param passwd:
        :param db:
        :param host:
        :param use_unicode:
        :param charset:
        :return:
        """
        self._conn = pymysql.connect(user, passwd, db, host, use_unicode, charset)
        self._cursor = self._conn.cursor()

    def execute(self, query, args=None):
        """
        Execute a query 并且提交,返回查询结果条数
        :return:
        """
        n = self._cursor.execute(query, args)
        self._conn.commit()
        return n

    def excuteMany(self, query, args):
        """
        Run several data against one query  并且提交
        cursor.executemany("insert into tb7(user,pass,licnese)values(%s,%s,%s)",
            [("u3","u3pass","11113"),("u4","u4pass","22224")])
        :param query: query to execute on server
        :param args: Sequence of sequences or mappings.  It is used as parameter.
        :return: Number of rows affected, if any.
        """
        n = self._cursor.executemany(query, args)
        self._conn.commit()
        return n

    def close(self):
        self._cursor.close()
        self._conn.close()
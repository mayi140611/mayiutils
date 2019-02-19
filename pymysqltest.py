#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: pymysqltest.py
@time: 2019/2/19 18:06
"""
from mayiutils.db.pymysql_wrapper import PyMysqlWrapper


if __name__ == '__main__':
    pmw = PyMysqlWrapper(host='10.1.192.118')
    pmw.execute('SELECT s.TY_NAME FROM sms_main s ')
    data = pmw._cursor.fetchall()
    print(data)
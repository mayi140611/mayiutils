#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: pymysqltest.py
@time: 2019/2/19 18:06
"""
from mayiutils.db.pymysql_wrapper import PyMysqlWrapper
from mayiutils.db.pymongo_wrapper import PyMongoWrapper
from mayiutils.pickle_wrapper import PickleWrapper as pkw
from mayiutils.filesystem.fileoperation_wrapper import FileOperationWrapper as fow
from mayiutils.nlp.jieba_wrapper import JieBaWrapper as jbw

if __name__ == '__main__':
    # keyworddict = dict()#用于存储查询数据库的关键字
    # typedict = dict()#
    #
    # pmw = PyMysqlWrapper(host='10.1.192.118')
    # table = 'sk_es_ref'
    #
    # pmw.execute('SELECT * FROM {} s '.format(table))
    # data = pmw._cursor.fetchall()
    #
    # for i in data:
    #     for ii in [2,3,4]:
    #         if i[ii]:
    #             if i[ii].strip() == '刘志明':
    #                 print(i[ii], len(i[ii]))
    #             keyworddict[i[ii].strip()] = i[1].strip()
    #             typedict[i[ii].strip()] = i[5].strip()
    #
    # pmw = PyMongoWrapper('h1')
    # dbname = 'jiankang39'
    # tablename = 'zznew'
    # table = pmw.getCollection(dbname, tablename)
    # datalist = list()
    # for t in pmw.findAll(table, fieldlist=['症状名称', '别名']):
    #     if '症状名称' in t:
    #         keyworddict[t['症状名称']] = t['症状名称']
    #         typedict[t['症状名称']] = 'zz'
    #     if '别名' in t:
    #         keyworddict[t['别名']] = t['症状名称']
    #         typedict[t['别名']] = 'zz'
    #
    # pkw.dump2File(keyworddict, 'tmp/keyworddict.pkl')
    # pkw.dump2File(typedict, 'tmp/typedict.pkl')




    typedict = pkw.loadFromFile('tmp/typedict.pkl')
    # jiebalist = ['{} 3\n'.format(i) for i in list(typedict.keys())]
    # fow.writeList2File(jiebalist, 'tmp/jiebadict.txt')
    # print(jbw.lcut('鼻腔癌该怎么治', HMM=False))
    jbw.loadUserDict('tmp/jiebadict.txt')
    print(jbw.lcut('鼻腔癌该怎么治', HMM=False))
    r = jbw.lcut('鼻腔癌该怎么治', HMM=False)
    r = jbw.lcut('刘志民在哪个医院', HMM=False)
    r = jbw.lcut('刘志民擅长治疗鼻咽癌么', HMM=False)
    r = jbw.lcut('刘志民在十院吗，我最近老是失眠', HMM=False)
    print(r)
    for i in r:
        if i in typedict:
            print(i, typedict[i])



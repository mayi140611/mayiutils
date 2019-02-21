#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: pymongotest.py
@time: 2019/1/21 16:11
"""
from mayiutils.db.pymongo_wrapper import PyMongoWrapper
from mayiutils.nlp.re_wrapper import ReWrapper as rew
import pandas as pd

if __name__ == '__main__':
    pmw = PyMongoWrapper('h1')
    # dbname = 'yyw'
    # tablename = 'drugdetail'
    dbname = 'jiankang39'
    tablename = 'zznew'
    table = pmw.getCollection(dbname, tablename)
    dbname = 'mybk'
    tablename = 'jbproductdetail'
    table2 = pmw.getCollection(dbname, tablename)
    list1 = list(pmw.findAll(table2, {'isdisease':'true'}, ['title']))
    list2 = [i['title'] for i in list1]
    # pattern1 = rew.getPattern(r'（')
    # pattern2 = rew.getPattern(r'）')
    # pattern3 = rew.getPattern(r'-')
    for i in pmw.findAll(table, fieldlist=['_id', '相关疾病']):
        if '相关疾病' in i:
            print(len(set(list2).intersection(i['相关疾病'])), len(i['相关疾病']))
            pmw.updateDoc(table, {'_id': i['_id']}, {'相关疾病1': list(set(list2).intersection(i['相关疾病']))})
        # r = rew.sub(pattern1, '(', i['通用名称'])
        # r = rew.sub(pattern2, ')', r)
        # r = rew.sub(pattern3, '-', r)
        # if r != i['通用名称']:
        #     pmw.updateDoc(table, {'_id': i['_id']}, {'通用名称': r})
        #     print(i)
        # print(len(i.keys()))
        # ii = pmw.findOne(table1, {'title': i['title']})
        # if ii and '别称' in ii:
        #     pmw.updateDoc(table2, {'_id': i['_id']}, {'别称': ii['别称']})
        # break
    #导出名医百科网疾病名称
    # l1 = list(pmw.findAll(table2, {'isdisease': 'true'}, ['title']))
    # l2 = [i['title'] for i in l1]
    # df = pd.DataFrame({'疾病名称': l2})
    # df.to_csv('名医百科疾病名称列表.csv', encoding='gbk')
    # print(l2[:5])



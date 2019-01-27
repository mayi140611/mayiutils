#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: pymongotest.py
@time: 2019/1/21 16:11
"""
from mayiutils.pymongo_wrapper import PyMongoWrapper
import pandas as pd

if __name__ == '__main__':
    pmw = PyMongoWrapper('h1')
    # table1 = pmw.getCollection('baidubaike', 'symptomsdetail')
    table2 = pmw.getCollection('mybk', 'jbproductdetail')
    # for i in pmw.findAll(table2, {'isdisease': 'true'}, ['_id', 'title']):
    #     print(i)
    #     ii = pmw.findOne(table1, {'title': i['title']})
    #     if ii and '别称' in ii:
    #         pmw.updateDoc(table2, {'_id': i['_id']}, {'别称': ii['别称']})
        # break
    #导出名医百科网疾病名称
    l1 = list(pmw.findAll(table2, {'isdisease': 'true'}, ['title']))
    l2 = [i['title'] for i in l1]
    df = pd.DataFrame({'疾病名称': l2})
    df.to_csv('名医百科疾病名称列表.csv', encoding='gbk')
    # print(l2[:5])

#!/usr/bin/python
# encoding: utf-8

from mayiutils.nlp.pypinyin_wrapper import PyPinyinWrapper
from mayiutils.db.pymongo_wrapper import PyMongoWrapper

if __name__ == "__main__":

    ppw = PyPinyinWrapper()
    # print(ppw.initialsOfWord("下雨天"))
    # print(ppw.initialsOfWord("四价HPV疫苗"))
    pmw = PyMongoWrapper('h1')
    table1 = pmw.getCollection('mybk', 'jbproductdetail')
    #增加拼音首字母字段
    # for i in pmw.findAll(table1, fieldlist=['_id', 'title']):
    #     print(i)
    #     titlepy = ppw.initialsOfWord(i["title"])
    #     pmw.updateDoc(table1, {'_id': i['_id']}, {'titlepy': titlepy})
    #增加fields字段
    ff = ['概述','临床表现','检查','诊断','治疗']

    for i in pmw.findAll(table1, {"isdisease": "true"}):
        print(i)
        l1 = list(set(i.keys()).intersection(set(ff)))
        # break
        pmw.updateDoc(table1, {'_id': i['_id']}, {'fileds': l1})

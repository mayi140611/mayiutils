#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: pypinyin_wrapper.py
@time: 2019/1/21 10:59

https://pypi.org/project/pypinyin/
"""
from pypinyin import Style, pinyin, lazy_pinyin
from mayiutils.db.pymongo_wrapper import PyMongoWrapper as pmw


class PyPinyinWrapper:

    @classmethod
    def initialsOfWord(cls, word):
        '''
        把汉字词语转换为拼音首字母缩写
        :param word:
        :return: 拼音首字母缩写
        '''
        return ''.join([ii[0] for ii in pinyin(word, style=Style.FIRST_LETTER, strict=False)])
    
    @classmethod
    def lazyPinyin(cls, s1):
        '''
        汉字短语转化为拼音列表，如'中心'，['zhong', 'xin']
        '''
        return lazy_pinyin(s1)


if __name__ == '__main__':
    mode = 2
    if mode == 2:
        """
        给MongoDB库中title添加首字母
        """
        table1 = pmw.getCollection('mybk', 'jbproductdetail')
        # 增加拼音首字母字段
        # for i in pmw.findAll(table1, fieldlist=['_id', 'title']):
        #     print(i)
        #     titlepy = ppw.initialsOfWord(i["title"])
        #     pmw.updateDoc(table1, {'_id': i['_id']}, {'titlepy': titlepy})
        # 增加fields字段
        ff = ['概述', '临床表现', '检查', '诊断', '治疗']

        for i in pmw.findAll(table1, {"isdisease": "true"}):
            print(i)
            l1 = list(set(i.keys()).intersection(set(ff)))
            # break
            pmw.updateDoc(table1, {'_id': i['_id']}, {'fileds': l1})
    if mode == 1:
        print(PyPinyinWrapper.initialsOfWord('二甲双胍'))#ejsg
        print(PyPinyinWrapper.lazyPinyin('二甲双胍'))#['er', 'jia', 'shuang', 'gua']
        print(PyPinyinWrapper.lazyPinyin('否极泰来'))#['pi', 'ji', 'tai', 'lai']
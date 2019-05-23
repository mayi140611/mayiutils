#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: dict_wrapper.py
@time: 2019/3/8 13:47
"""

class DictWrapper:
    """
    无序的对象集合
    字典当中的元素是通过键来存取的，而不是通过偏移存取

    """
    @classmethod
    def genDict(cls, keylist, valuelist):
        """
        字典由索引(key)和它对应的值value组成
        已知两个等长的list生成dict，一个list的元素作为key，另一个list的元素作为value

        :param keylist:
        :param valuelist:
        :return:
        """
        return dict(zip(keylist, valuelist))

    @classmethod
    def getValue(cls, dict1, key, defaultVal=None):
        """
        获取dict1的key对应的值, 如果key不存在，返回默认值
        :param dict1:
        :param key:
        :param defaultVal:
        :return:
        """
        return dict1.get(key, defaultVal)

    @classmethod
    def getItems(cls, dict1):
        """
        返回[(key1, v1),....]
        :param dict1:
        :return:
        """
        return dict1.items()

    @classmethod
    def getKeys(cls, dict1):
        """
        dict_keys([2, 'one'])
        :param dict1:
        :return:
        """
        return dict1.keys()

    @classmethod
    def getValues(cls, dict1):
        """
        dict_values(['This is two', 'This is one'])
        :param dict1:
        :return:
        """
        return dict1.values()

    @classmethod
    def update(cls, dict1, dict2):
        """
        用dict2的key-value，更新dict1的key-value
        :param dict1:
        :param dict2:
        :return:
        """
        dict1.update(dict2)
        return dict1

    @classmethod
    def deleteKey(cls, dict1, key):
        """
        删除dict1的key
        :param dict1:
        :param key:
        :return:
        """
        if key in dict1:
            del dict1[key]
        return dict1

    @classmethod
    def clear(cls, dict1):
        """
        清空dict1
        :param dict1:
        :return:
        """
        dict1.clear()
        return dict1


if __name__ == '__main__':

    mode = 3
    if mode == 3:
        d = {i: 0 for i in 'sbme'}#字典生成式
        print(d)
        print(len(d))
    if mode == 2:
        list1 = list(range(5))
        dict1 = dict()
        for i, q in enumerate(list1, start=1):
            print(i, q)
            dict1[q] = i
        print(dict1)
    if mode == 1:
        """
        字典遍历
        """
        D = {'x': 1, 'y': 2, 'z': 3}
        for key in D:
            print(key, '=>', D[key])
        for key, value in D.items():  # 方法二
            print(key, '=>', value)
        for value in D.values():  # 方法四
            print(value)
        for i, v in enumerate(D):
            print(i, v)
        """
        0 x
        1 y
        2 z
        """
    if mode == 0:
        keylist = ['a', 'b', 'c']
        valuelist = list(range(3))
        d = DictWrapper.genDict(keylist, valuelist)
        d2 = {'c': 8, 'd': 9}
        print(DictWrapper.update(d, d2))#{'a': 0, 'b': 1, 'c': 8, 'd': 9}
        print(DictWrapper.clear(d))#{}
        print(DictWrapper.getItems(d))#dict_items([('a', 0), ('b', 1), ('c', 2)])
        print(DictWrapper.getValue(d, 'd', '空值'))
        print(DictWrapper.deleteKey(d, 'b'))
        print(DictWrapper.deleteKey(d, 'b'))


#!/usr/bin/python
# encoding: utf-8


class SetWrapper(object):
    """
    主要是对python中的集合set的相关操作的封装
    set() 可变集合，可以在集合中增加删除元素
    frozenset() 不可变集合

    #赋值操作 类似于 +=
    s1 |= s2 或 s1.update(s2)#{1, 2, 3, 4, 5, 6}
    s1 -= s2 或 s1.difference_update(s2)
    s1 &= s2 或 s1.intersection_update(s2)
    """
    def __init__(self):
        pass
    
    @classmethod
    def add(cls, set1, element):
        """
        Add an element to a set。注意只能一个一个元素添加
        :param set1:
        :param element:
        :return:
        """
        return set1.add(element)

    @classmethod
    def discard(cls, set1, element):
        """
        删除几何中的一个元素。对于不存在的元素，不会报异常
        :param set1:
        :param element:
        :return:
        """
        return set1.discard(element)

    @classmethod
    def remove(cls, set1, element):
        """
        删除几何中的一个元素。对于不存在的元素，会报异常
        :param set1:
        :param element:
        :return:
        """
        return set1.remove(element)

    @classmethod
    def union(cls, set1, set2):
        """
        并集
        s1 | s2 或者s1.union(s2)
        :param set1:
        :param set2:
        :return:
        """
        return set1.union(set2)

    @classmethod
    def intersection(cls, set1, set2):
        """
        交集
        s1 & s2 或 s1.intersection(s2)
        :param set1:
        :param set2:
        :return:
        """
        return set1.intersection(set2)

    @classmethod
    def isdisjoint(cls, set1, set2):
        """
        是否有交集
        s1.isdisjoint(s2)#没有交集返回True
        :param set1:
        :param set2:
        :return:
        """
        return set1.isdisjoint(set2)
    @classmethod
    def difference(cls, set1, set2):
        """
        差集
        s1 - s2 或 s1.difference(s2)
        :param set1:
        :param set2:
        :return:
        """
        return set1.difference(set2)

    @classmethod
    def symmetric_difference(cls, set1, set2):
        """
        对称差集（交集的补集）
        s1 = set([1,2,3,4,3,2,1])
        s2 = set([3,4,5,6])
        s1 ^ s2 或 s1.symmetric_difference(s2)#{1, 2, 5, 6}
        :param set1:
        :param set2:
        :return:
        """
        return set1.symmetric_difference(set2)

    @classmethod
    def issubset(cls, set1, set2):
        """
        子集（被包含）
        s1 < s2 或 s1.issubset(s2)
        :param set1:
        :param set2:
        :return:
        """
        return set1.issubset(set2)

    @classmethod
    def issuperset(cls, set1, set2):
        """
        超集（包含）
        s1>set([1,2,1]) 或 s1.issuperset([1,2,1])
        :param set1:
        :param set2:
        :return:
        """
        return set1.issuperset(set2)


if __name__ == '__main__':
    mode = 2
    if mode == 2:
        print(set('我是中国人，我爱中国'))#{'人', '中', '国', '，', '爱', '我', '是'}
    if mode == 1:
        print(set(['失眠', '乏力', '多梦', '疲劳']) == set(['失眠', '多梦', '疲劳', '乏力']))#True
        print(set(['失眠', '乏力', '多梦', '疲劳']) == ['失眠', '多梦', '疲劳', '乏力'])#False
        print(set(['失眠', '多梦', '疲劳', '乏力']).intersection(set(['失眠', '多梦', '疲劳','发热'])))#{'失眠', '多梦', '疲劳'}
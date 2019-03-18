#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: itertools_wrapper.py
@time: 2019/3/19 4:07
"""
import itertools


if __name__ == '__main__':
    mode = 3
    list1 = [1, 2, 3, 4]
    if mode == 5:
        """
        repeat
        """
        print(list(itertools.repeat([1, 2], 5)))#[[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]
    if mode == 4:
        """
        chain, 链接一组迭代器
        """
        for i in itertools.chain(itertools.combinations(list1, 2), itertools.permutations(list1, 2)):
            print(i)
    if mode == 3:
        """
        笛卡尔积，两两配对组合
        """
        list2 = ['a', 'b', 'c']
        for i in itertools.product(list1, list2):
            print(i)
        print(list(itertools.product([1, 2], [3, 4], [5, 6])))
    if mode == 2:
        """
        组合
        """
        for i in itertools.combinations(list1, 2):
            print(i)
    if mode == 1:
        """
        排列
        """

        print(itertools.permutations(list1, 2))#<itertools.permutations object at 0x000001806B1CBA98>
        for i in itertools.permutations(list1, 2):
            print(i)








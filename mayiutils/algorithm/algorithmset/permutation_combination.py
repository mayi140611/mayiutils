#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: permutation_combination.py
@time: 2019/3/23 13:27
排列组合
"""
from scipy.special import comb, perm
from itertools import combinations, permutations

def permutation(n, m):
    """
    n中选择m个数进行排列
    先总结出递推公式
    :param n:
    :param m:
    :return:
    """
    return 1 if m == 0 else permutation(n, m - 1) * (n - m + 1)


if __name__ == "__main__":
    print(permutation(5, 4))#120
    print(perm(5, 4))
    print(comb(5, 4))
    #调用 itertools 获取排列组合的全部情况数
    print(list(permutations([1, 2, 3], 2)))
    print(list(combinations([1, 2, 3], 2)))
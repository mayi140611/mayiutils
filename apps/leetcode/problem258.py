#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: problem258.py
@time: 2019/3/21 13:59
"""
class Solution(object):
    """
    给定一个非负整数 num，反复将各个位上的数字相加，直到结果为一位数。

    示例:

    输入: 38
    输出: 2
    解释: 各位相加的过程为：3 + 8 = 11, 1 + 1 = 2。 由于 2 是一位数，所以返回 2。
    """
    def addDigits(self, num):
        """
        40 ms	10.8 MB
        :type num: int
        :rtype: int
        """
        while num / 10 >= 1:
            num_str = str(num)
            num = sum([int(i) for i in num_str])
        return num

    def addDigits1(self, num):
        """
        其他人的做法，结果是正确的。为什么是取余？
        执行用时 : 36 ms, 在Add Digits的Python提交中击败了30.00% 的用户
        内存消耗 : 10.7 MB, 在Add Digits的Python提交中击败了0.00% 的用户
        :param num:
        :return:
        """
        if num == 0:
            return 0
        elif num%9 == 0:
            return 9
        else:
            return num%9
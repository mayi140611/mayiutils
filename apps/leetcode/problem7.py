#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: problem7.py
@time: 2019/3/26 15:52
"""


class Solution(object):
    """
    https://leetcode-cn.com/problems/reverse-integer/
    给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。

    示例 1:

    输入: 123
    输出: 321
     示例 2:

    输入: -123
    输出: -321
    """
    def reverse(self, x):
        """
        执行用时 : 36 ms, 在Reverse Integer的Python提交中击败了93.45% 的用户
        内存消耗 : 11.8 MB, 在Reverse Integer的Python提交中击败了2.37% 的用户
        :type x: int
        :rtype: int
        """
        x = str(x)
        flag = 1
        if x[0] == '-':
            flag = -1
            x = x[1:]
        x = flag * int(x[::-1])
        if x > 2**31-1 or x < - 2**31:
            x = 0
        return x
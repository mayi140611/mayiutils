#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: problem5.py
@time: 2019/3/24 20:53
"""


class Solution(object):
    """
    https://leetcode-cn.com/problems/longest-palindromic-substring/
    给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。
    回文串就是正着读和反着读都一样的字符串。
    示例 1：

    输入: "babad"
    输出: "bab"
    注意: "aba" 也是一个有效答案。
    """
    def longestPalindrome(self, s):
        """
        超出时间限制!!!
        :type s: str
        :rtype: str
        """
        if len(s) == 0:
            return ''
        a = s[0]
        for i in range(2, len(s)+1):
            for ii in range(0, len(s)-i+1):
                ss = s[ii: (i+ii)]
                t = len(ss)//2
                if ss[:t] == ss[::-1][:t]:
                    a = ss
                    break
        return a
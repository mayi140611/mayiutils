#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: problem3.py
@time: 2019/3/21 10:37
"""


class Solution(object):
    """
    https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/
    给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。

    示例 1:

    输入: "abcabcbb"
    输出: 3
    解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

    """
    def lengthOfLongestSubstring(self, s):
        """
        执行用时 : 480 ms, 在Longest Substring Without Repeating Characters的Python提交中击败了12.21% 的用户
        内存消耗 : 12.7 MB, 在Longest Substring Without Repeating Characters的Python提交中击败了1.11% 的用户
        :type s: str
        :rtype: int

        思路：set的长度 == list的长度
        换种思路，从小向大找
        """
        flag = 0
        for i in range(len(s)):
            for ii in range(len(s)-i):
                if len(set(s[ii: (ii+i+1)])) == (i+1):
                    flag = i+1
                    break
            if flag < i+1:
                break
        return flag
    def lengthOfLongestSubstring1(self, s):
        """
        :type s: str
        :rtype: int

        思路1：set的长度 == list的长度
        问题：测试结果中，有一个超级长的字符串，时间超出限制
        """
        for i in range(len(s)):
            if i == 0:
                if len(set(s)) == len(s):
                    return len(s)
            for ii in range(i + 1):
                ss = s[ii: (ii + (len(s) - i))]
                if len(set(ss)) == len(ss):
                    return len(ss)
        return 0
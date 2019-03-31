#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: problem10.py
@time: 2019/3/31 23:07
"""


class Solution(object):
    """
    https://leetcode-cn.com/problems/regular-expression-matching/
    自己没有做出来，
    看评论可以使用递归，或者更优的动态规划！
    """
    def isMatch(self, s, p):
        """
        递归， 别人的方法，其实没太看懂。。。
        执行用时 : 1704 ms, 在Regular Expression Matching的Python提交中击败了13.13% 的用户
        内存消耗 : 11.8 MB, 在Regular Expression Matching的Python提交中击败了0.00% 的用户
        :type s: str
        :type p: str
        :rtype: bool
        """

        if len(p) == 0:
            return len(s) == 0

        if len(p) >= 2 and '*' == p[1]:
            return (self.isMatch(s, p[2:])) or not (len(s) == 0) and (s[0] == p[0] or '.' == p[0]) and self.isMatch(
                s[1:], p)
        else:
            return (not (len(s) == 0)) and (s[0] == p[0] or '.' == p[0]) and self.isMatch(s[1:], p[1:])





if __name__ == '__main__':
    s = Solution()
    r = s.isMatch('abcabc', '.*abc.*abcc*.*')
    print(r)
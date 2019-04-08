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
    10. 正则表达式匹配
    给定一个字符串 (s) 和一个字符模式 (p)。实现支持 '.' 和 '*' 的正则表达式匹配。

    '.' 匹配任意单个字符。
    '*' 匹配零个或多个前面的元素。
    匹配应该覆盖整个字符串 (s) ，而不是部分字符串。

    说明:

    s 可能为空，且只包含从 a-z 的小写字母。
    p 可能为空，且只包含从 a-z 的小写字母，以及字符 . 和 *。
    示例 1:

    输入:
    s = "aa"
    p = "a"
    输出: false
    解释: "a" 无法匹配 "aa" 整个字符串。

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
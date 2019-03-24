#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: problem1015.py
@time: 2019/3/23 13:23
"""
class Solution(object):
    """
    https://leetcode-cn.com/problems/numbers-with-repeated-digits/
    给定正整数 N，返回小于等于 N 且具有至少 1 位重复数字的正整数的个数。
    示例 1：

    输入：20
    输出：1
    解释：具有至少 1 位重复数字的正数（<= 20）只有 11 。
    示例 2：

    输入：100
    输出：10
    解释：具有至少 1 位重复数字的正数（<= 100）有 11，22，33，44，55，66，77，88，99 和 100 。
    示例 3：

    输入：1000
    输出：262
    """
    def numDupDigitsAtMostN(self, N):
        """
        执行用时 : 32 ms, 在Numbers With Repeated Digits的Python提交中击败了100.00% 的用户
        内存消耗 : 11.9 MB, 在Numbers With Repeated Digits的Python提交中击败了100.00% 的用户
        https://leetcode.com/problems/numbers-with-repeated-digits/discuss/256725/JavaPython-Count-the-Number-Without-Repeated-Digit
        :type N: int
        :rtype: int
        """
        L = map(int, str(N + 1))
        res, n = 0, len(L)

        def A(m, n):
            return 1 if n == 0 else A(m, n - 1) * (m - n + 1)

        for i in range(1, n): res += 9 * A(9, i - 1)
        s = set()
        for i, x in enumerate(L):
            for y in range(0 if i else 1, x):
                if y not in s:
                    res += A(9 - i, n - i - 1)
            if x in s: break
            s.add(x)
        return N - res
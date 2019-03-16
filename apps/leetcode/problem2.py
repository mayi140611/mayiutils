#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: problem2.py
@time: 2019/3/16 12:44
"""


# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    """
    https://leetcode-cn.com/problems/add-two-numbers/
    给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。

    如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。

    您可以假设除了数字 0 之外，这两个数都不会以 0 开头。

    示例：

    输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
    输出：7 -> 0 -> 8
    原因：342 + 465 = 807
    """


    def addTwoNumbers1(self, l1, l2):
        """
        执行用时 : 100 ms, 在Add Two Numbers的Python提交中击败了18.53% 的用户
        内存消耗 : 10.9 MB, 在Add Two Numbers的Python提交中击败了0.86% 的用户
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        l11, l22 = [], []
        flag1, flag2 = 0, 0
        # 把两个listnode对象转换为list
        while True:
            if flag1 == 0:
                l11.append(l1.val)
                if not l1.next:
                    flag1 = 1
                else:
                    l1 = l1.next
            if flag2 == 0:
                l22.append(l2.val)
                if not l2.next:
                    flag2 = 1
                else:
                    l2 = l2.next
            if flag1 + flag2 == 2:
                break
        num1 = len(l11)
        num2 = len(l22)
        if num1 < num2:
            l11.extend([0] * (num2 - num1))
        elif num2 < num1:
            l22.extend([0] * (num1 - num2))
        result = []
        flag = 0
        for i in range(max(num1, num2)):
            r = l11[i] + l22[i] + flag
            if r > 9:
                r -= 10
                flag = 1
            else:
                flag = 0
            result.append(r)
        if flag == 1:
            result.append(flag)
        return result

    def addTwoNumbers2(self, l1, l2):
        """
        改进， 不再把l1和l2转化为list
        执行用时 : 176 ms, 在Add Two Numbers的Python提交中击败了1.79% 的用户
        内存消耗 : 10.8 MB, 在Add Two Numbers的Python提交中击败了0.86% 的用户
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        num1, num2 = 0, 0
        flag1, flag2 = 0, 0
        l11 = l1
        l22 = l2
        while True:
            if flag1 == 0:
                num1 += 1
                if not l1.next:
                    flag1 = 1
                else:
                    l1 = l1.next
            if flag2 == 0:
                num2 += 1
                if not l2.next:
                    flag2 = 1
                else:
                    l2 = l2.next
            if flag1 + flag2 == 2:
                break
        result = []
        flag = 0
        for i in range(max(num1, num2)):
            r = l11.val + l22.val + flag
            if r > 9:
                r -= 10
                flag = 1
            else:
                flag = 0
            result.append(r)
            if not l11.next:
                l11.next = ListNode(0)
            if not l22.next:
                l22.next = ListNode(0)
            l11 = l11.next
            l22 = l22.next
        if flag == 1:
            result.append(flag)
        return result


    @classmethod
    def addTwoNumbers(self, l1, l2):
        """
        执行用时 : 104 ms, 在Add Two Numbers的Python提交中击败了15.54% 的用户
        内存消耗 : 10.6 MB, 在Add Two Numbers的Python提交中击败了0.86% 的用户
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        result = []
        flag = 0
        flag1, flag2 = 0, 0
        while True:
            r = l1.val + l2.val + flag
            if r > 9:
                r -= 10
                flag = 1
            else:
                flag = 0
            result.append(r)
            if not l1.next:
                flag1 = 1
                l1.next = ListNode(0)
            if not l2.next:
                flag2 = 1
                l2.next = ListNode(0)
            if flag1 + flag2 == 2:
                break
            l1 = l1.next
            l2 = l2.next
        if flag == 1:
            result.append(flag)
        return result

if __name__ == "__main__":
    a = [2, 4, 3]
    b = [5, 6, 4]
    l1 = ListNode(2)
    l1.next = ListNode(4)
    l1.next.next = ListNode(3)
    l2 = ListNode(5)
    l2.next = ListNode(6)
    l2.next.next = ListNode(4)
    print(Solution.addTwoNumbers(l1, l2))
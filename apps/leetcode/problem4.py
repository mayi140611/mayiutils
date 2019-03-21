#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: problem4.py
@time: 2019/3/21 13:40
"""


class Solution(object):
    """
    https://leetcode-cn.com/problems/median-of-two-sorted-arrays/
    给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。

    请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。

    你可以假设 nums1 和 nums2 不会同时为空。

    示例 1:

    nums1 = [1, 3]
    nums2 = [2]

    则中位数是 2.0
    示例 2:

    nums1 = [1, 2]
    nums2 = [3, 4]

    则中位数是 (2 + 3)/2 = 2.5
    """
    def findMedianSortedArrays(self, nums1, nums2):
        """
        执行用时 : 64 ms, 在Median of Two Sorted Arrays的Python提交中击败了93.66% 的用户
        内存消耗 : 11 MB, 在Median of Two Sorted Arrays的Python提交中击败了3.82% 的用户
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        nums1.extend(nums2)
        nums1 = sorted(nums1)

        index1 = index2 = len(nums1) // 2
        if len(nums1) % 2 == 0:
            index2 -= 1
        return (nums1[index1] + nums1[index2]) / 2.0
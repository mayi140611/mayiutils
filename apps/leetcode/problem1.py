#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: problem1.py
@time: 2019/3/16 11:19
"""
class Solution:
    """
    https://leetcode-cn.com/problems/two-sum/
    给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。

    你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

    示例:

    给定 nums = [2, 7, 11, 15], target = 9

    因为 nums[0] + nums[1] = 2 + 7 = 9
    所以返回 [0, 1]

    初看题目觉得很简单，实际做起来发现也不简单，还是挺有收获
    """
    @classmethod
    def twoSum(self, nums, target):
        """
        最优解
        执行用时 : 28 ms, 在Two Sum的Python提交中击败了98.36% 的用户
        内存消耗 : 11.7 MB, 在Two Sum的Python提交中击败了0.96% 的用户
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        hashmap = {}
        for index, num in enumerate(nums):
            another_num = target - num
            if another_num in hashmap:
                return [hashmap[another_num], index]
            hashmap[num] = index
        return None
    @classmethod
    def twoSum3(self, nums, target):
        """
        暴力法：
        时间复杂度：O(n^2)
        空间复杂度：O(1)O(1)

        实际测试时，说是执行超时了。。。
        :param nums:
        :param target:
        :return:
        """
        for i in range(len(nums)):
            for ii in range(len(nums)):
                if ii == i:
                    continue
                if nums[i] + nums[ii] == target:
                    return i, ii
    @classmethod
    def twoSum1(cls, nums, target):
        """
        错解，如果测试用例是 [3, 2, 4] 6时，不能通过，原因是6-3=3
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        d1 = {}
        for i, v in enumerate(nums):
            d1[v] = i
        for i in range(len(nums)):
            temp = (target - nums[i])
            if temp in d1:
                return i, d1[temp]
    @classmethod
    def twoSum2(cls, nums, target):
        """
        错解，
        在twoSum1上的改进。如果测试用例是 [3, 3] 6时，不能通过，原因是nums中有两个重复的元素
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        d1 = {}
        for i, v in enumerate(nums):
            d1[v] = i
        for i in range(len(nums)):
            temp = (target - nums[i])
            if temp != nums[i] and temp in d1:
                return i, d1[temp]


if __name__ == "__main__":
    nums = [3,2,4]
    target = 6
    print(Solution.twoSum1(nums, target))
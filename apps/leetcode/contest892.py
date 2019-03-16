#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: contest892.py
@time: 2019/3/16 15:02
"""
import numpy as np


class Solution(object):
    """
    892. 三维形体的表面积
    在 N * N 的网格上，我们放置一些 1 * 1 * 1  的立方体。

    每个值 v = grid[i][j] 表示 v 个正方体叠放在单元格 (i, j) 上。

    返回结果形体的总表面积。
    """
    @classmethod
    def surfaceArea(cls, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        思路：先算出表面总数，然后依次在x, y, z方向减去重叠的面
        """
        arr = np.array(grid)
        totalsum = np.sum(grid) * 6
        # 减去z方向的重叠面
        sub = 0
        for a in arr.flat:
            if a != 0:
                sub += (a - 1) * 2
        # 减去y方向的重叠面
        for i in range(arr.shape[1]):
            for ii in range(arr.shape[0] - 1):
                sub += int(np.min([arr[ii + 1, i], arr[ii, i]]) * 2)

        # 减去y方向的重叠面
        for i in range(arr.shape[0]):
            for ii in range(arr.shape[1] - 1):
                sub += int(np.min([arr[i, ii + 1], arr[i, ii]]) * 2)
        return int(totalsum - sub)

if __name__ == "__main__":
    a = [[1,1,1],[1,0,1],[1,1,1]]
    print(Solution.surfaceArea(a))
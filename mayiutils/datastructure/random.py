#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: random.py
@time: 2019/3/4 15:27
"""
import random
import numpy as np


class Random:

    @classmethod
    def randomInt(cls, a, b):
        """
        Return a random integer N such that a <= N <= b. Alias for randrange(a, b+1).
        :param a:
        :param b:
        :return:
        """
        return random.randint(a, b)


if __name__ == '__main__':
    mode = 3
    if mode == 3:
        """
        生成各种分布的随机数
        """
        #Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
        print(np.random.rand(5))#[0.05657772 0.45262035 0.71823949 0.54420319 0.43150836]
        print(np.random.rand(2, 3))
        """
        [[0.62938819 0.31702677 0.54956982]
        [0.29557825 0.41988991 0.84604464]]
        """
        print(np.random.rand(2, 2, 3))
        """
        [[[0.13143365 0.75426488 0.68169651]
          [0.94488093 0.07218573 0.19677621]]
        
         [[0.81995996 0.79454814 0.35505484]
          [0.93745205 0.00294314 0.97878135]]]
        """
    if mode == 2:
        """
        设置随机种子seed的方法
        random.seed()
        numpy.random.seed()不是线程安全的，
        如果程序中有多个线程最好使用numpy.random.RandomState实例对象来创建
        或者使用random.seed()来设置相同的随机数种子。
        注意这几种方式不能交叉使用，
        如使用了random.seed()设置了种子，那么就要用random模块的方法生成随机数
        """
        random.seed(14)
        """
        最佳的的方式，在每次随机过程时，都设置一次种子
        """
        print(random.sample([1, 3, 5, 7], 2))#Used for random sampling without replacement(无放回采样）
        random.seed(14)
        print(random.sample(range(100), 2))#[13, 78]
        random.seed(14)
        print(random.sample(range(100), 2))#[13, 78]
        print('hello world')
        random.seed(14)
        print(random.sample(range(100), 2))#[13, 78]
        random.seed(14)
        print(random.sample(range(100), 2))#[13, 78]
        print('-----------------------------')
        np.random.seed(14)
        print(np.random.randint(0, 100, (2, )))#[88 12]
        print(np.random.RandomState(14).randint(0, 100, (2, )))#[88 12]
    if mode == 1:
        """
        产生随机整数
        """
        r = Random.randomInt(0, 3)#可能的值：0,1,2,3
        print(r)
        r = Random.randomInt(0, 0)#可能的值：0
        print(r)
        print(np.random.randint(0, 100, (5, )))#[24 12 52 69 53]
#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: random.py
@time: 2019/3/4 15:27
"""
import random
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
    r = Random.randomInt(0, 0)#可能的值：0,1,2,3
    print(r)
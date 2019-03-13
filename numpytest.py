#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: numpytest.py
@time: 2019/2/20 11:01
"""
from mayiutils.datastructure.numpy_wrapper import NumpyWrapper as npw
import numpy as np

if __name__ == "__main__":
    arr1 = npw.buildArrayFromArrayList([[1,2]])
    arr2 = npw.buildArrayFromArrayList([[3,4]])
    # npw.saveTxt(arr, 'tmp/arr.txt')
    # arr = npw.loadTxt('tmp/arr.txt')
    # np.savez('tmp/a.npz', arr1=arr1, arr2=arr2)
    # print(arr)
    # a = np.load('tmp/a.npz')
    # print(a['arr1'], a['arr2'])
    # np.savez_compressed('tmp/a_compressed.npz', arr1=arr1, arr2=arr2)
    a = np.load('tmp/a_compressed.npz')
    print(a['arr1'], a['arr2'])

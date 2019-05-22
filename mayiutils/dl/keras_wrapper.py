#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: keras_wrapper.py
@time: 2019/2/24 9:25
"""
from keras import backend as K
import numpy as np


if __name__ == "__main__":
    x = np.arange(8).reshape((2, 4))
    print(x)
    """
    [[0 1 2 3]
    [4 5 6 7]]
    """
    print(np.expand_dims(x, 2))
    """
    [[[0]
      [1]
      [2]
      [3]]
    
     [[4]
      [5]
      [6]
      [7]]]
    """
    print(K.expand_dims(x, 2))#Tensor("ExpandDims:0", shape=(2, 4, 1), dtype=int32)
    print(K.one_hot(x, 10))#Tensor("one_hot:0", shape=(2, 4, 10), dtype=float32)
    # Permutes(交换) axes in a tensor.
    print(K.permute_dimensions(x, (1, 0)))#Tensor("transpose:0", shape=(4, 2), dtype=int32)
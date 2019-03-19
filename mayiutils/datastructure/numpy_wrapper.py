#!/usr/bin/python
# encoding: utf-8

import numpy as np


class NumpyWrapper(object):
    def __init__(self):
        pass

    # '''
    # ---------------------------------------------------------------------------------
    # 生成ndarray对象
    # ---------------------------------------------------------------------------------
    # '''
    @classmethod
    def arange(self, start, stop, step=1, dtype=None):
        '''
        生成1D ndarray
        Return evenly spaced values within a given interval.
        '''
        return np.arange(start, stop, step, dtype)

    @classmethod
    def linspace(self, start, stop, num=50, endpoint=True, retstep=False, dtype=None):
        '''
        生成1D ndarray
        Return evenly spaced numbers over a specified interval.
        print(np.linspace(0, 5, 6))#[0. 1. 2. 3. 4. 5.]
        '''
        return np.linspace(start, stop, num, endpoint, retstep, dtype)

    @classmethod
    def buildArrayFromArrayList(cls, arraylist, dtype=np.int32):
        """

        :param arraylist:
            [[1, 2], [3, 4]],  [1, 2, 3, 4]
        :return:
        """
        return np.array(arraylist, dtype)

    @classmethod
    def buildZerosArray(cls, shape, dtype=np.float, order='C'):
        """
        构建全0矩阵
        :param shape:
        :param dtype:
        :param order:
        :return:
        """
        return np.zeros(shape,dtype,order)

    @classmethod
    def buildOnesArray(cls, shape, dtype=np.float, order='C'):
        """
        构建全1矩阵
        :param shape:
        :param dtype:
        :param order:
        :return:
        """
        return np.ones(shape, dtype, order)

    @classmethod
    def buildEmptyArray(cls, shape, dtype=np.float, order='C'):
        """
        Return a new array of given shape and type, without initializing entries
        empty, unlike zeros, does not set the array values to zero, and may therefore be marginally faster.
        On the other hand, it requires the user to manually set all the values in the array, and should be used with caution.
        :param shape:
        :param dtype:
        :param order:
        :return:
        """
        return np.empty(shape, dtype, order)

    @classmethod
    def buildEye(cls, N, M=None, k=0, dtype=float, order='C'):
        """
        单位矩阵
        Return a 2-D array with ones on the diagonal and zeros elsewhere.
        :param shape:
        :param dtype:
        :param order:
        :return:
        """
        return np.fromfunction().eye(N, M, k, dtype, order)

    @classmethod
    def buildFromFunction(cls, function, shape, **kwargs):
        """
        Construct an array by executing a function over each coordinate.
        The resulting array therefore has a value fn(x, y, z) at coordinate (x, y, z).
        :param function:
        :param shape:
        :param kwargs:
        :return:
        """
        return np.fromfunction(function, shape, **kwargs)

    @classmethod
    def addNewAxisLast(cls, matr):
        '''
        在matr的最后加一个维度
        @matr: 1D ndarray
        '''
        return matr[:, np.newaxis]

    @classmethod
    def flatten(cls, order='C'):
        '''
        Return a copy of the array collapsed(坍塌) into one dimension.
        @order: {'C', 'F', 'A', 'K'}, optional
        'C' means to flatten in row-major (C-style) order.
        'F' means to flatten in column-major (Fortran-
        style) order. 'A' means to flatten in column-major
        order if `a` is Fortran *contiguous* in memory,
        row-major order otherwise. 'K' means to flatten
        `a` in the order the elements occur in memory.
        The default is 'C'.
        '''
        return np.flatten(order)
    # ---------------------------------------------------------------------------------
    # 描述性统计
    # ---------------------------------------------------------------------------------
    
    @classmethod
    def max(cls, a, axis=None):
        '''
        求最大值
        @axis: None表示求整个数组的最大值；0表示求每列最大值；1表示求每行最大值
        '''
        return np.max(a, axis)
    @classmethod
    def min(cls, a, axis=None):
        '''
        求最大值
        @axis: None表示求整个数组的最大值；0表示求每列最大值；1表示求每行最大值
        '''
        return np.min(a, axis)
    @classmethod
    def sum(self, a, axis=None):
        '''
        求最大值
        @axis: None表示求整个数组的最大值；0表示求每列最大值；1表示求每行最大值
        '''
        return np.sum(a, axis)
    @classmethod
    def mean(self, a, axis=None):
        '''
        求最大值
        @axis: None表示求整个数组的最大值；0表示求每列最大值；1表示求每行最大值
        '''
        return np.mean(a, axis)
    # ---------------------------------------------------------------------------------
    # 生成随机数
    # ---------------------------------------------------------------------------------
    @classmethod
    def generate_random_seed(self, seed=None):
        '''
        但是numpy.random.seed()不是线程安全的，
        如果程序中有多个线程最好使用numpy.random.RandomState实例对象来创建
        或者使用random.seed()来设置相同的随机数种子。
                
        import random
        random.seed(1234567890)
        a = random.sample(range(10),5) 

        注意： 随机数种子seed只有一次有效，在下一次调用产生随机数函数前没有设置seed，则还是产生随机数。
        '''
        return np.random.RandomState(seed)
    @classmethod
    def uniform_rand(self, *param, seed=None):
        '''
        Create an array of the given shape and populate it with
        random samples from a uniform distribution
        over ``[0, 1)``.
        '''
        return self.generate_random_seed(seed).rand(*param)
    @classmethod
    def uniform_randint(self, low, high=None, size=None, dtype='l', seed=None):
        '''
        Return random integers from `low` (inclusive) to `high` (exclusive).
        '''
        return self.generate_random_seed(seed).randint(low, high, size, dtype)
    @classmethod
    def randn(self, *param, seed=None):
        '''
        Return a sample (or samples) from the "standard normal" distribution.
        '''
        return self.generate_random_seed(seed).randn(*param)
    # '''
    # ---------------------------------------------------------------------------------
    # 线性代数
    # ---------------------------------------------------------------------------------
    # '''

    @classmethod
    def inv(self, matr):
        '''
        求解矩阵的逆
        '''
        return np.linalg.inv(matr)
    @classmethod
    def det(self, matr):
        '''
        计算行列式
        '''
        return np.linalg.det(matr)
    @classmethod
    def sum(self, matr, axis=None, dtype=None, out=None):
        return np.sum(matr, axis, dtype, out)

    # '''
    # ---------------------------------------------------------------------------------
    # ndarray合并
    # ---------------------------------------------------------------------------------
    # '''
    @classmethod
    def vstack(cls, tup):
        """
        列向叠加ndarray
        :param tup: 由ndarray对象组成的元组
            (sequence of ndarrays) The arrays must have the same shape along all but the first axis.
            1-D arrays must have the same length.
        :return:
        """
        return np.vstack(tup)
    # '''
    # ---------------------------------------------------------------------------------
    # 文件保存和读取
    # ---------------------------------------------------------------------------------
    # '''
    @classmethod
    def save(cls, array, filepath):
        """
        默认情况下，数组以未压缩的原始二进制格式保存在扩展名为npy的文件中
        :param array:
        :param filepath: 't.npy'
        :return:
        """
        return np.save(filepath, array)

    @classmethod
    def load(cls, filepath):
        """
        加载save保存的npy文件
        :param filepath:
        :return:
        """
        return np.load(filepath)

    @classmethod
    def saveTxt(cls, array, filepath):
        """
        将1D/2D数组写入以某种分隔符隔开的文本文件中, 注意保存三维以上的数据会报错
        :param array:
        :param filepath: 't.txt'
        :return:
        """
        return np.savetxt(filepath, array)

    @classmethod
    def loadTxt(cls, filepath):
        """
        加载savetxt保存的txt文件
        :param filepath:
        :return:
        """
        return np.loadtxt(filepath)
    @classmethod
    def savez(cls, array, filepath):
        """
        Save several arrays into a single file in uncompressed .npz format.
        示例函数，
            arr1 = npw.buildArrayFromArrayList([[1,2]])
            arr2 = npw.buildArrayFromArrayList([[3,4]])
            np.savez('tmp/a.npz', arr1=arr1, arr2=arr2) 或者压缩格式
            np.savez_compressed('tmp/a_compressed.npz', arr1=arr1, arr2=arr2)
            a = np.load('tmp/a.npz')
            print(a['arr1'], a['arr2'])
        :param array:
        :param filepath: 't.txt'
        :return:
        """
        return np.savez(filepath, array)

    @classmethod
    def shuffle(cls, arr):
        """
        就地shuffle打乱数据
        Modify a sequence in-place by shuffling its contents.
        This function only shuffles the array along the first axis of a multi-dimensional array. The order of sub-arrays is changed but their contents remains the same.
        三行代码打乱样本数据：
        >>> permutation = np.random.permutation(train_label.shape[0])
        >>> shuffled_dataset = train_data[permutation, :, :]
        >>> shuffled_labels = train_label[permutation]

        示例
        >>> arr = np.arange(10)
        >>> np.random.shuffle(arr)
        >>> arr
        [1 7 5 2 9 4 3 6 0 8]
        Multi-dimensional arrays are only shuffled along the first axis:
        >>> arr = np.arange(9).reshape((3, 3))
        >>> np.random.shuffle(arr)
        >>> arr
        array([[3, 4, 5],
               [6, 7, 8],
               [0, 1, 2]]
        :param arr: array_like
        The array or list to be shuffled.
        :return:
        """
        return np.random.shuffle(arr)


if __name__ == '__main__':
    mode = 4
    if mode == 4:
        """
        扁平化
        """
        # np.meshgrid()
        x = y = np.arange(0, 10, 0.1)
        xx, yy = np.meshgrid(x, y)
        print(xx.shape)#(100, 100)
        print(xx.ravel())# [0.  0.1 0.2 ... 9.7 9.8 9.9]
        print(xx.ravel().shape)#(10000,)
        # np.c_ 两两配对
        print(np.c_[xx.ravel(), yy.ravel()].shape)# (10000, 2)
    if mode == 3:
        """
        扩展维度
        """
        arr = np.arange(4).reshape((2, 2))
        print(arr)
        """
        [[0 1]
        [2 3]]
        """
        print(np.expand_dims(arr, axis=0))
        """
         [[[0 1]
          [2 3]]]       
        """
        print(np.expand_dims(arr, axis=1))
        """
         [[[0 1]]
        
         [[2 3]]]       
        """
        print(np.expand_dims(arr, axis=-1))# 添加一维在数组的最后一维后面
        """
          [[[0]
          [1]]
        
         [[2]
          [3]]]
        """
        print(np.expand_dims(arr, axis=2))
        """
          [[[0]
          [1]]
        
         [[2]
          [3]]]
        """

    if mode == 2:
        """
        遍历numpy中的每一个元素
        """
        arr = np.arange(9).reshape(3, 3)
        print(arr)
        print(arr.flat)#<numpy.flatiter object at 0x000002803FC8BE10>
        print(list(arr.flat))#[0, 1, 2, 3, 4, 5, 6, 7, 8]
    if mode == 1:
        print(np.ones([2, ]))
        #测试dtype转换
        a = np.array(['我fadffs'], dtype=str)
        print(a.dtype, a.ndim, a)#<U7 1 ['我fadffs']
        b = np.arange(16).reshape((-1, 2, 2, 2))
        print(b)
        print(b.ndim)#4 维度 dimension
    if mode == 0:
        # a = NumpyWrapper.arange(1, 5)#[1 2 3 4]
        # a = NumpyWrapper.linspace(1, 5, num=5)#[1. 2. 3. 4. 5.]
        a = NumpyWrapper.buildEmptyArray([2,3])
        b = NumpyWrapper.buildZerosArray([2,3])
        # a = NumpyWrapper.vstack((a, b))
        # b = NumpyWrapper.addNewAxisLast(a)
        b = np.array([b])
        a = np.array([a])
        b = NumpyWrapper.vstack((a, b))
        print(b)
        print(a)

        print(a == None)
        print(None == None)
        arr1 = np.arange(16).reshape(8, 2)
        arr = np.arange(8)
        #按照第一个维度打乱
        np.random.shuffle(arr)
        print(arr)
        permutation = np.random.permutation(arr.shape[0])
        print(type(permutation), permutation)
        b = arr[permutation, :]
        print(b)
        #不能是list对象，会报错。只能是ndarray对象
        print(arr[list(permutation), :])
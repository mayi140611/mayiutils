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
    mode = 10
    """
    ndarray creation
    """
    # 生成1D ndarray
    # np.arange(start, stop, step, dtype)
    # print(np.arange(1, 9, 2, dtype=np.float))#[[1. 3. 5. 7.]
    # print(np.arange(1, 9, 2))#[1 3 5 7]
    arr1d = np.arange(9)
    print(arr1d)# [0 1 2 3 4 5 6 7 8]

    #np.linspace(start, stop, num, endpoint, retstep, dtype)
    # print(np.linspace(1, 9, 5))# [1. 3. 5. 7. 9.]

    # Return a 2-D array with ones on the diagonal and zeros elsewhere.
    eye = np.eye(3)
    print(eye)
    """
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
    """
    print(np.diag(np.arange(1, 4)))
    """
[[1 0 0]
 [0 2 0]
 [0 0 3]]
    """
    # 生成2D ndarray

    """
    selection
    """
    # 随机从arr1d中采样
    a = np.random.choice(arr1d, 2)
    print(a)# [8 3]
    """
    a : 1-D array-like or int
                If an ndarray, a random sample is generated from its elements.
                If an int, the random sample is generated as if a were np.arange(a)
    """

    
    """
    合并
    """
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
    print(np.concatenate((a, b), axis=0))
    """
[[1 2]
 [3 4]
 [5 6]]
    """
    print(np.concatenate((a, b.T), axis=1))
    """
[[1 2 5]
 [3 4 6]]
    """
    print(np.concatenate((a, b), axis=None))
    """
[1 2 3 4 5 6]  
    """
    if mode == 10:
        """
        线性代数 linalg
        """
        a = np.arange(48).reshape((12, 4))
        #svd分解
        U, s, Vh = np.linalg.svd(a, full_matrices=True)
        print(U.shape)#(12, 12)
        print(s)#[1.88956663e+02 3.92167338e+00 1.34838487e-14 9.73158362e-16]
        print(Vh)
        """
[[-0.47596788 -0.49178085 -0.50759382 -0.52340679]
 [-0.68808035 -0.24114641  0.20578753  0.65272148]
 [ 0.38232588 -0.21743345 -0.71211076  0.54721832]
 [ 0.39220775 -0.80791256  0.43920186 -0.02349705]]
        """
        s_matr = np.concatenate((np.diag(s), np.zeros((U.shape[0]-s.shape[0], s.shape[0]))), axis=0)
        print(U.dot(s_matr.dot(Vh)))
        """
[[2.22044605e-16 1.00000000e+00 2.00000000e+00 3.00000000e+00]
 [4.00000000e+00 5.00000000e+00 6.00000000e+00 7.00000000e+00]
 [8.00000000e+00 9.00000000e+00 1.00000000e+01 1.10000000e+01]
 [1.20000000e+01 1.30000000e+01 1.40000000e+01 1.50000000e+01]
 [1.60000000e+01 1.70000000e+01 1.80000000e+01 1.90000000e+01]
 [2.00000000e+01 2.10000000e+01 2.20000000e+01 2.30000000e+01]
 [2.40000000e+01 2.50000000e+01 2.60000000e+01 2.70000000e+01]
 [2.80000000e+01 2.90000000e+01 3.00000000e+01 3.10000000e+01]
 [3.20000000e+01 3.30000000e+01 3.40000000e+01 3.50000000e+01]
 [3.60000000e+01 3.70000000e+01 3.80000000e+01 3.90000000e+01]
 [4.00000000e+01 4.10000000e+01 4.20000000e+01 4.30000000e+01]
 [4.40000000e+01 4.50000000e+01 4.60000000e+01 4.70000000e+01]]
        """
    if mode == 9:
        """
        各种数学运算
        """
        a = np.arange(4)
        print(a)
        print(np.sqrt(a))
        print(np.square(a))
        a = np.ones((2, 2))
        print(a - np.arange(2))
        """
    [[1. 0.]
     [1. 0.]]
        """
    if mode == 8:
        a = np.arange(8).reshape((2, 2, 2))
        print(a)
        """
        [[[0 1]
          [2 3]]
        
         [[4 5]
          [6 7]]]
        """
        print(np.dot(a, a))
        """
        If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
        If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.
        If either a or b is 0-D (scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred.
        If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
        If a is an N-D array and b is an M-D array (where M>=2), 
            it is a sum product over the last axis of a and the second-to-last(倒数第二个) axis of b
            也就是说点积发生在a,b矩阵最后两个维度上
        [[[[ 2  3]
           [ 6  7]]
        
          [[ 6 11]
           [26 31]]]
        
        
         [[[10 19]
           [46 55]]
        
          [[14 27]
           [66 79]]]]
        """
        print(a[:, :, 1])
    if mode == 7:
        """
        * 
        np.greater
        """
        a = np.arange(8).reshape((2, 4))
        b = np.arange(2).reshape((2, 1))
        print(a, b)
        """
        [[0 1 2 3]
         [4 5 6 7]] 
         [[0]
         [1]]
        """
        print(b * a)
        """
        [[0 0 0 0]
        [4 5 6 7]]
        """
        print(a * b)
        """
        [[0 0 0 0]
        [4 5 6 7]]
        """
        print(np.greater(a, 0))
        """
        [[False  True  True  True]
        [ True  True  True  True]]
        """
    if mode == 6:
        """
        keepdims
        """
        a = np.arange(15).reshape((3, 5))

        print(np.sum(a, 1))#[10 35 60]
        print(np.sum(a, 1, keepdims=True))
        """
        [[10]
         [35]
         [60]]
        """
    if mode == 5:
        """
        利用numpy生成one-hot数组
        """
        test_labels = [1, 2, 3, 4, 5, 6, 7]
        adata = np.array(test_labels)

        # print(adata[:, None])#相当于给最后增加一维
        def make_one_hot(data1):
            return (np.arange(10) == data1[:, None]).astype(np.integer)

        my_one_hot = make_one_hot(adata)
        print(my_one_hot)
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

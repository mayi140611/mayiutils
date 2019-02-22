#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: preparedata.py
@time: 2019/2/22 13:52

# A script to load images and make batch.
"""
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mayiutils.filesystem.os_wrapper import OsWrapper as osw
import pydicom

FLAGS = dict()
FLAGS['width'] = 256
FLAGS['height'] = 256
FLAGS['depth'] = 40 # 3
batch_index = 0
filenames = []

FLAGS['data_dir'] = '/home/ikbeom/Desktop/DL/MNIST_simpleCNN/data'
FLAGS['num_class'] = 4

filedir = 'D:/Desktop/DF/train_set'
maxSlicesNum = 120

def getMaxSlicesNum(filepath):
    """
    获取最大的切片数
    :param filepath:
    :return:
    """
    list1 = osw.listDir(filepath)
    num = 0
    for i in list1:
        num1 = len(osw.listDir(osw.join(filepath, i)))
        if num < num1:
            num = num1
    print('最大切片数：{}'.format(num))
    return num

def test(filepath):
    list1 = osw.listDir(filepath)
    for i in list1:
        print(osw.join(filepath, i))
        dcmDir = osw.listDir(osw.join(filepath, i))#每一组dcm存放的目录
        num = len(dcmDir)
        print(num)
        if num < maxSlicesNum:
            pre = (maxSlicesNum-num) // 2
            destArr = np.zeros((pre, 512, 512))
        else:
            destArr = np.array([-1])
        for ii in osw.listDir(osw.join(filepath, i)):
            ds = pydicom.read_file(osw.join(filepath,i,ii))  # 读取.dcm文件
            img = ds.pixel_array  # 提取图像信息
            if np.all(destArr==-1):
                destArr = np.array([img])
            else:
                destArr = np.vstack((destArr, np.array([img])))
            # break
        if destArr.shape[0] < maxSlicesNum:
            destArr = np.vstack((destArr, np.zeros(((maxSlicesNum-destArr.shape[0]), 512, 512))))
        print(destArr.shape)

        # break
    # print()


if __name__ == "__main__":
    print(FLAGS)
    test(filedir)
    # getMaxSlicesNum(filedir)#120

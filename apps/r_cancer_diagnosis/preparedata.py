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
import pandas as pd
import matplotlib.pyplot as plt
from mayiutils.filesystem.os_wrapper import OsWrapper as osw
from mayiutils.pickle_wrapper import PickleWrapper as pkw
import pydicom
import os
import cv2
import math
import sys

FLAGS = dict()
FLAGS['width'] = 256
FLAGS['height'] = 256
FLAGS['depth'] = 40 # 3
batch_index = 0
filenames = []

FLAGS['data_dir'] = '/home/ikbeom/Desktop/DL/MNIST_simpleCNN/data'
FLAGS['num_class'] = 4

maxSlicesNum = 587#train1和train2中最大的切片数量


def getMaxSlicesNum(filepath):
    """
    获取最大的切片数
    :param filepath:
    :return:
    """
    num = 0
    for f in ['train1_label.csv', 'train2_label.csv']:
        df = pd.read_csv(osw.join(filepath, f))
        for line in df.itertuples():
            c = len(osw.listDir(osw.join(filepath, f.split('_')[0], line[1])))
            if num < c:
                num = c
    print('最大切片数：{}'.format(num))
    return num

def prepareTrainData(filepath):
    """
    准备训练数据，含train_set, train_label

    :param filepath:
    :return:
    """
    train_set = list()
    train_label = list()
    count0 = 0
    count1 = 0
    count2 = 0
    for f in ['train1_label.csv', 'train2_label.csv']:
        df = pd.read_csv(osw.join(filepath, f))

        for line in df.itertuples():
            print('{}_{}'.format(count0, line[1]))
            count0 += 1
            num = len(osw.listDir(osw.join(filepath, f.split('_')[0]+'_jpg', line[1])))
            destArr = -1
            if num < maxSlicesNum:
                # destArr = np.zeros(((maxSlicesNum - num) // 2, 64, 64))
                flag = 0
            else:
                # destArr = np.array([-1])
                flag = 1
            for i in osw.listDir(osw.join(filepath, f.split('_')[0]+'_jpg', line[1])):
                # img = pydicom.read_file(i).pixel_array
                img = cv2.imread(osw.join(filepath, f.split('_')[0]+'_jpg', line[1], i), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (64, 64))
                if flag == 0:
                    destArr = np.array([img])
                else:
                    destArr = np.vstack((np.zeros(((maxSlicesNum - num) // 2, 64, 64)), np.array([img])))
            if destArr.shape[0] < maxSlicesNum:
                destArr = np.vstack((destArr, np.zeros(((maxSlicesNum - destArr.shape[0]), 64, 64))))
            print(destArr.shape)
            train_set.append(destArr)
            train_label.append(line[2])
            if count1 == 199:
                np.savez_compressed('batchs/trainsets_batch{}.npz'.format(count2), train_set=np.array(train_set), train_label=np.array(train_label))
                count2 += 1
                count1 = 0
                train_label = list()
                train_set = list()
            else:
                count1 += 1
    if train_set:
        print(len(train_set))
        np.savez_compressed('batchs/trainsets_batch{}.npz'.format(count2), train_set=np.array(train_set),
                            train_label=np.array(train_label))
    # list1 = osw.listDir(filepath)
    # for i in list1:
    #     print(osw.join(filepath, i))
    #     dcmDir = osw.listDir(osw.join(filepath, i))#每一组dcm存放的目录
    #     num = len(dcmDir)
    #     print(num)
    #     if num < maxSlicesNum:
    #         pre = (maxSlicesNum-num) // 2
    #         destArr = np.zeros((pre, 512, 512))
    #     else:
    #         destArr = np.array([-1])
    #     for ii in osw.listDir(osw.join(filepath, i)):
    #         ds = pydicom.read_file(osw.join(filepath,i,ii))  # 读取.dcm文件
    #         img = ds.pixel_array  # 提取图像信息
    #         if np.all(destArr==-1):
    #             destArr = np.array([img])
    #         else:
    #             destArr = np.vstack((destArr, np.array([img])))
    #         # break
    #     if destArr.shape[0] < maxSlicesNum:
    #         destArr = np.vstack((destArr, np.zeros(((maxSlicesNum-destArr.shape[0]), 512, 512))))
    #     print(destArr.shape)

        # break
    # print()

def prepareTestData(filepath):
    """
    准备测试数据，成为模型可以直接预测的格式

    :param filepath:
    :return:
    """
    train_set = list()
    train_label = list()
    count0 = 0
    count1 = 0
    count2 = 0
    for f in ['test.csv']:
        df = pd.read_csv(osw.join(filepath, f))

        for line in df.itertuples():
            print('{}_{}'.format(count0, line[1]))
            train_label.append(line[1])
            count0 += 1
            num = len(osw.listDir(osw.join(filepath, 'test_jpg', line[1])))
            destArr = -1
            if num < maxSlicesNum:
                # destArr = np.zeros(((maxSlicesNum - num) // 2, 64, 64))
                flag = 0
            else:
                # destArr = np.array([-1])
                flag = 1
            for i in osw.listDir(osw.join(filepath, 'test_jpg', line[1])):
                # img = pydicom.read_file(i).pixel_array
                img = cv2.imread(osw.join(filepath, 'test_jpg', line[1], i), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (64, 64))
                if flag == 0:
                    destArr = np.array([img])
                else:
                    destArr = np.vstack((np.zeros(((maxSlicesNum - num) // 2, 64, 64)), np.array([img])))
            if destArr.shape[0] < maxSlicesNum:
                destArr = np.vstack((destArr, np.zeros(((maxSlicesNum - destArr.shape[0]), 64, 64))))
            elif destArr.shape[0] > maxSlicesNum:
                destArr = destArr[:587, :, :]
            print(destArr.shape)
            train_set.append(destArr)
            if count1 == 199:
                np.savez_compressed('test_batchs/testsets_batch{}.npz'.format(count2), test_set=np.array(train_set),
                                    testimgname=np.array(train_label))
                count2 += 1
                count1 = 0
                train_label = list()
                train_set = list()
            else:
                count1 += 1
    if train_set:
        print(len(train_set))
        np.savez_compressed('test_batchs/testsets_batch{}.npz'.format(count2), test_set=np.array(train_set), testimgname=np.array(train_label))

def get_files(trainDir='D:/Desktop/pets', ratio=0.7):
    topdir = trainDir
    imgpathlist = list()
    labellist = list()
    labeldict = dict()
    num = 0
    count = 0
    for dirpath, dirs, files in os.walk(topdir):
        for f in sorted(files):
            # print(os.path.join(dirpath, f))
            labelname = f.split('_')[0]
            if labelname not in labeldict.values():
                if num > 4: break
                labeldict[num] = labelname
                num += 1
                count = 0
            if count < 100:
                imgpathlist.append(os.path.join(dirpath, f))
                labellist.append(num)
                count += 1

    print(len(labellist), len(imgpathlist), labeldict)
    seq = list(range(len(imgpathlist)))
    np.random.shuffle(seq)
    n_train = math.floor(len(imgpathlist)*ratio)
    n_val = len(imgpathlist) - n_train
    # print(seq[:5])
    train_images = list()
    val_images = list()
    train_labels = list()
    val_labels = list()
    for t in seq[:n_train]:
        i = imgpathlist[0]
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        # img2 = cv2.imread(i)
        # img = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        # img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # img4 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        # print(img.shape, img1.shape, img2.shape, img3.shape, img4.shape)
        # print(img, img1)
        # break
        # #把图片的大小统一为512*512
        # if img.shape[0] < 512:
        #     arr = np.zeros((512-img.shape[0], img.shape[1]))
        #     img = np.vstack((img, arr))
        # elif img.shape[0] > 512:
        #     img = img[:512, :]
        # if img.shape[1] < 512:
        #     arr = np.zeros((img.shape[0], 512-img.shape[1]))
        #     img = np.hstack((img, arr))
        # elif img.shape[1] > 512:
        #     img = img[:, :512]
        train_images.append(img)
        train_labels.append(labellist[t])
    for t in seq[n_train:]:
        i = imgpathlist[t]
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        # print(img.shape)
        #把图片的大小统一为512*512
        # if img.shape[0] < 512:
        #     arr = np.zeros((512-img.shape[0], img.shape[1]))
        #     img = np.vstack((img, arr))
        # elif img.shape[0] > 512:
        #     img = img[:512, :]
        # if img.shape[1] < 512:
        #     arr = np.zeros((img.shape[0], 512-img.shape[1]))
        #     img = np.hstack((img, arr))
        # elif img.shape[1] > 512:
        #     img = img[:, :512]
        val_images.append(img)
        val_labels.append(labellist[t])
    return np.array(train_images), np.array(train_labels), np.array(val_images), np.array(val_labels)


if __name__ == "__main__":
    print(FLAGS)
    # test(filedir)
    # X_train, y_train, X_test, y_test = get_files()
    # np.savez_compressed('littleCBIRdatasets.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    filedir = sys.argv[1]
    # prepareTrainData(filedir)
    prepareTestData(filedir)
    # filedir = 'D:/Desktop/DF'
    # n = getMaxSlicesNum(filedir)#587
    # print(n)
    # print(np.all(np.array([-1])==-1))



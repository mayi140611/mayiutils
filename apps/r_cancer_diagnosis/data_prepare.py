#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: data_prepare.py
@time: 2019/3/6 12:47

准备训练集、验证集和测试集
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.data import Dataset
import tensorflow as tf

class DataPrepare:

    def __init__(self, maxSliceNum=192, imageSize=48, totalNum=7574, trainsetNum=6800, baseDir='/home/ian/datafoundation'):
        self._baseDir = baseDir
        self._totalNum = totalNum
        self._trainsetNum = trainsetNum
        self._maxSliceNum = maxSliceNum
        self.imagesize = imageSize

    def _parse_function(self, filenames, label):
        imageshape = (self.imagesize, self.imagesize)
        count = 0
        img = list()
        for i in range(192):
            filename = filenames[i]
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string)
            image_resized = tf.image.resize_images(image_decoded, imageshape)
            if count == 0:
                img = tf.reshape(image_resized, (-1, imageshape[0], imageshape[1], 1))
                count += 1
            else:
                img = tf.concat([img, tf.reshape(image_resized, (-1, imageshape[0], imageshape[1], 1))], 0)

        return img / 255.0, tf.cast(label, tf.float32)

    def change_img(self, x, y):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        x = tf.image.random_crop(x, [self._maxSliceNum, int(self.imagesize*0.9), int(self.imagesize*0.9), 1])  # 注意用法
        print(x)
        x = tf.image.pad_to_bounding_box(x, 1, 1, self.imagesize, self.imagesize)#补0 offset_height, offset_width, target_height, target_width
        print(x)
        return x, y

    def prepareTrainandValImagePath(self):
        """
        准备图片路径。格式如下：
        imagePathlist =
        [
            [file1,..., filen],
            [file1,..., filen],
            ...
        ]
        注意每个img的厚度是不同的，导致每个的n值都不同，这里的n是一个超参数，需要要统一（多则删除，不足补充
        labellist = [
            [1],
            [1],
            [1],
            [1],
        ]
        :param n: 每个肝癌样本的切片数
        :return:
        """
        imagePathlist = list()
        labellist = list()
        for csv in ['train1_label.csv', 'train2_label.csv']:
            df = pd.read_csv(os.path.join(self._baseDir, csv))
            for line in df.itertuples():
                imgdir = os.path.join(self._baseDir, csv.split('_')[0]+'_jpg', line[1])
                files = [os.path.join(imgdir, i)for i in os.listdir(imgdir)]
                if len(files) > self._maxSliceNum:
                    files = files[:self._maxSliceNum]
                elif len(files) < self._maxSliceNum:
                    files.extend([os.path.join(self._baseDir, 'white.jpg')]*(self._maxSliceNum-len(files)))
                imagePathlist.append(files)
                labellist.append([line[2]])
        print(len(imagePathlist), len(labellist))
        return np.array(imagePathlist), np.array(labellist)

    def prepareTestImagePath(self):
        """
        准备图片路径。格式如下：
        imagePathlist =
        [
            [file1,..., filen],
            [file1,..., filen],
            ...
        ]
        注意每个img的厚度是不同的，导致每个的n值都不同，这里的n是一个超参数，需要要统一（多则删除，不足补充

        :return:
        """
        imagePathlist = list()
        for csv in ['test.csv']:
            df = pd.read_csv(os.path.join(self._baseDir, csv))
            for line in df.itertuples():
                imgdir = os.path.join(self._baseDir, 'test_jpg', line[1])
                files = [os.path.join(imgdir, i) for i in os.listdir(imgdir)]
                if len(files) > self._maxSliceNum:
                    files = files[:self._maxSliceNum]
                elif len(files) < self._maxSliceNum:
                    files.extend([os.path.join(self._baseDir, 'white.jpg')]*(self._maxSliceNum-len(files)))
                imagePathlist.append(files)
        print(len(imagePathlist))
        return np.array(imagePathlist)

    def prepareTrainDataSet(self, imagePatharr, labelarr):
        """

        :param imagePatharr:
        :param labelarr:
        :param ratio:
        :return:
        """
        dataset1 = Dataset.from_tensor_slices((imagePatharr[:self._trainsetNum], labelarr[:self._trainsetNum]))
        # dataset2 = dataset1.map(_parse_function).shuffle(buffer_size=20).repeat(100).map(change_img).batch(2)
        dataset2 = dataset1.map(self._parse_function).shuffle(buffer_size=20).map(self.change_img).repeat(100).batch(2)
        dataset2 = dataset2.prefetch(2)
        return dataset2

    def prepareValDataset(self, imagePatharr, labelarr):
        dataset1 = Dataset.from_tensor_slices((imagePatharr[self._trainsetNum:], labelarr[self._trainsetNum:]))
        dataset2 = dataset1.map(self._parse_function).repeat(100000000).batch(2)
        dataset2 = dataset2.prefetch(2)
        return dataset2

    def prepareTestDataset(self, imagePatharr):
        dataset1 = Dataset.from_tensor_slices((imagePatharr, np.zeros((imagePatharr.shape[0], 1))))
        dataset2 = dataset1.map(self._parse_function).batch(5)
        dataset2 = dataset2.prefetch(5)
        return dataset2

if __name__ == '__main__':
    dp = DataPrepare()
    # dp.prepareTrainandValImagePath()
    dp.prepareTestImagePath()



#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: tfmodel2.py
@time: 2019/3/1 9:30
"""
import os
import sys
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

imagesize = 72
imagesize_cropped = 64

def change_img(x, y):
    i = random.randint(0, 3)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    print(x)
    x = tf.image.random_crop(x, [128, 64, 64, 1])#注意用法
    print(x)
    # if i == 0:
    #     x = tf.image.random_flip_left_right(x)
    # elif i == 1:
    #     x = tf.image.random_flip_up_down(x)
    # elif i == 2:
    #     x = tf.image.random_crop(x, 64)
        # x = tf.image.resize_image_with_pad(imagesize, imagesize)

    return x, y
def _parse_function(filenames, label, imageshape=(imagesize, imagesize)):
    count = 0
    img = list()
    for i in range(128):
    # for filename in filenames:
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

def _parse_function1(filenames, imageshape=(imagesize, imagesize)):
    count = 0
    img = list()
    for i in range(128):
    # for filename in filenames:
        filename = filenames[i]
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, imageshape)
        if count == 0:
            img = tf.reshape(image_resized, (-1, imageshape[0], imageshape[1], 1))
            count += 1
        else:
            img = tf.concat([img, tf.reshape(image_resized, (-1, imageshape[0], imageshape[1], 1))], 0)

    return img / 255.0

class TfKerasModel:
    """

    """
    def __init__(self, maxSliceNum=128, totalNum=7574, trainsetNum=6800, baseDir='/home/ian/datafoundation'):
        self._baseDir = baseDir
        self._totalNum = totalNum
        self._trainsetNum = trainsetNum
        self._maxSliceNum = maxSliceNum

    def countN(self):
        """
        统计每个肝癌样本的切片数，画出评率直方图
        :return:
        """
        countlist = list()
        for csv in ['train1_label.csv', 'train2_label.csv']:
            df = pd.read_csv(os.path.join(self._baseDir, csv))
            for line in df.itertuples():
                imgdir = os.path.join(self._baseDir, csv.split('_')[0]+'_jpg', line[1])
                count = len(os.listdir(imgdir))
                countlist.append(count)
        s = pd.Series(countlist).value_counts().sort_values(ascending=False)
        print(s[:25])
        """
        40     571
        42     4imagesize
        39     460
        41     445
        37     388
        43     372
        38     345
        45     341
        46     282
        44     262
        47     222
        36     199
        48     181
        """
        countlist = list()
        for csv in ['test.csv']:
            df = pd.read_csv(os.path.join(self._baseDir, csv))
            for line in df.itertuples():
                imgdir = os.path.join(self._baseDir, 'test_jpg', line[1])
                count = len(os.listdir(imgdir))
                countlist.append(count)
        s = pd.Series(countlist).value_counts().sort_values(ascending=False)
        print(s)

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
                files = [os.path.join(imgdir, i)for i in os.listdir(imgdir)]
                if len(files) > self._maxSliceNum:
                    files = files[:self._maxSliceNum]
                elif len(files) < self._maxSliceNum:
                    files.extend([os.path.join(self._baseDir, 'white.jpg')]*(self._maxSliceNum-len(files)))
                imagePathlist.append(files)
        return np.array(imagePathlist)

    def prepareTrainDataSet(self, imagePatharr, labelarr):
        """

        :param imagePatharr:
        :param labelarr:
        :param ratio:
        :return:
        """
        dataset1 = Dataset.from_tensor_slices((imagePatharr[:self._trainsetNum], labelarr[:self._trainsetNum]))
        # dataset2 = dataset1.map(_parse_function).shuffle(buffer_size=self._trainsetNum).repeat(10).batch(1)
        dataset2 = dataset1.map(_parse_function).shuffle(buffer_size=20).repeat(100).map(change_img).batch(2)
        # dataset2 = dataset2.prefetch(2)
        return dataset2

    def prepareValDataset(self, imagePatharr, labelarr):
        dataset1 = Dataset.from_tensor_slices((imagePatharr[self._trainsetNum:], labelarr[self._trainsetNum:]))
        dataset2 = dataset1.map(_parse_function).repeat(100000000).map(change_img).batch(1)
        # dataset2 = dataset2.prefetch(2)
        return dataset2

    def prepareTestDataset(self, imagePatharr):
        dataset1 = Dataset.from_tensor_slices((imagePatharr, np.zeros((imagePatharr.shape[0], 1))))
        dataset2 = dataset1.map(_parse_function).map(change_img).batch(5)
        # dataset2 = dataset2.prefetch(2)
        return dataset2

    def buildModel(self):
        model = tf.keras.Sequential()
        model.add(Conv3D(32, 5, strides=1, padding='same', activation='relu', input_shape=(self._maxSliceNum, imagesize, imagesize, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Conv3D(32, 5, strides=1, padding='same', activation='relu', input_shape=(self._maxSliceNum, imagesize, imagesize, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Conv3D(64, 5, strides=1, padding='same', activation='relu', input_shape=(self._maxSliceNum, imagesize, imagesize, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Conv3D(64, 5, strides=1, padding='same', activation='relu', input_shape=(self._maxSliceNum, imagesize, imagesize, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Flatten())
        adam = Adam(lr=1e-4)
        # model.add(Dense(1, activation='softmax'))
        # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def buildVgg11Model(self):
        """
        vgg16太大了，模仿vgg11的结构
        :return:
        """
        model = tf.keras.Sequential()
        model.add(Conv3D(64, 3, strides=1, padding='valid', activation='relu', input_shape=(self._maxSliceNum, imagesize_cropped, imagesize_cropped, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='valid'))
        model.add(Conv3D(128, 3, strides=1, padding='valid', activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='valid'))
        model.add(Conv3D(256, 3, strides=1, padding='valid', activation='relu'))
        model.add(Conv3D(256, 3, strides=1, padding='valid', activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='valid'))
        # model.add(Conv3D(512, 3, strides=1, padding='valid', activation='relu', input_shape=(self._maxSliceNum, imagesize, imagesize, 1)))
        # model.add(Conv3D(512, 3, strides=1, padding='valid', activation='relu', input_shape=(self._maxSliceNum, imagesize, imagesize, 1)))
        # model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='valid'))
        # model.add(Conv3D(512, 3, strides=1, padding='valid', activation='relu', input_shape=(self._maxSliceNum, imagesize, imagesize, 1)))
        model.add(Conv3D(512, 3, strides=1, padding='valid', activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='valid'))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        adam = Adam(lr=1e-4)
        # model.add(Dense(1, activation='softmax'))
        # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def loadModel(self, filename):
        return load_model(os.path.join(self._baseDir, 'result/models', filename))

    # def predict(self, model, testset):



if __name__ == '__main__':
    mode = sys.argv[1]
    # mode = '3'
    m = TfKerasModel()
    if mode == '4':

        model = m.buildVgg11Model()
        print(model.summary())
        print('训练模式')
        imagePatharr, labelarr = m.prepareTrainandValImagePath()
        # print(imagePatharr.shape, labelarr.shape)#(7574, 128) (7574, 1)
        trainDataset = m.prepareTrainDataSet(imagePatharr, labelarr)
        # print(trainDataset.output_types)  # ==> (tf.string, tf.int64)
        # print(trainDataset.output_shapes)  # ==> (TensorShape([Dimension(128)]), TensorShape([Dimension(1)]))
        valDataset = m.prepareValDataset(imagePatharr, labelarr)
        filepath1 = "result/models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        callbacks = [
            # Interrupt training if `val_loss` stops improving for over 2 epochs
            # tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            # Write TensorBoard logs to `./logs` directory
            tf.keras.callbacks.TensorBoard(log_dir='./result/log'),
            tf.keras.callbacks.ModelCheckpoint(filepath1, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', cooldown=0, min_lr=0)
        ]
        print('开启训练')
        model.fit(trainDataset, steps_per_epoch=34, epochs=10000, validation_data=valDataset, validation_steps=774, callbacks=callbacks)
        print('train finished!')
    if mode == '3':
        arr = np.load('D:/Desktop/DF/result.npy')
        r = list()
        for a in arr:
            if a < 0.5:
                r.append(0)
            else:
                r.append(1)
        print(arr.shape, arr[:3], r[:3])
        df = pd.read_csv('D:/Desktop/DF/test.csv')
        df['ret'] = r[:4027]
        df.to_csv('D:/Desktop/DF/test_predict6.csv', index=False)
    if mode == '2':
        """
        预测
        """
        print('预测模式')
        modelname = sys.argv[2]
        testPatharr = m.prepareTestImagePath()
        testDataset = m.prepareTestDataset(testPatharr)
        print('加载模型{}'.format(modelname))
        model = m.loadModel(modelname)
        # model = m.loadModel('weights-improvement-06-0.71.hdf5')
        # model.evaluate(testDataset, steps=806)
        print('开始预测')
        result = model.predict(testDataset, steps=806)
        print('预测完成')
        print(result.shape)
        np.save('result.npy', result)
    if mode == '1':
        """
        训练模型
        """
        print('训练模式')
        imagePatharr, labelarr = m.prepareTrainandValImagePath()
        # print(imagePatharr.shape, labelarr.shape)#(7574, 128) (7574, 1)
        trainDataset = m.prepareTrainDataSet(imagePatharr, labelarr)
        # print(trainDataset.output_types)  # ==> (tf.string, tf.int64)
        # print(trainDataset.output_shapes)  # ==> (TensorShape([Dimension(128)]), TensorShape([Dimension(1)]))
        valDataset = m.prepareValDataset(imagePatharr, labelarr)
        model = m.buildModel()
        print(model.summary())
        filepath1 = "result/models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        callbacks = [
            # Interrupt training if `val_loss` stops improving for over 2 epochs
            # tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            # Write TensorBoard logs to `./logs` directory
            tf.keras.callbacks.TensorBoard(log_dir='./result/log'),
            tf.keras.callbacks.ModelCheckpoint(filepath1, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', cooldown=0, min_lr=0)
        ]
        print('开启训练')
        model.fit(trainDataset, steps_per_epoch=68, epochs=5000, validation_data=valDataset, validation_steps=774, callbacks=callbacks)
        print('train finished!')
    if mode == '0':
        """
        统计图片个数
        """
        m.countN()
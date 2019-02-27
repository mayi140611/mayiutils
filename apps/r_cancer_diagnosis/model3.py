#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: model3.py
@time: 2019/2/27 14:48
"""
import pandas as pd
from mayiutils.filesystem.os_wrapper import OsWrapper as osw
import numpy as np
import cv2
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution3D,MaxPooling3D,Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

from keras.models import load_model
import sys

class Model:
    def loadValData(self, filepath, imgsize=128, maxSlicesNum=587, clip1=96, clip2=384):
        train_set = list()
        train_label = list()
        for f in ['train1_label.csv', 'train2_label.csv']:
            df = pd.read_csv(osw.join(filepath, f))
            count1 = 0
            for line in df.itertuples():
                # 每个csv的前150条数据不参加训练，留作验证集
                if count1 == 150:
                    break
                print('{}_{}'.format(f, count1))
                count1 += 1
                imagepath1 = osw.join(filepath, f.split('_')[0]+'_jpg', line[1])
                num = len(osw.listDir(imagepath1))
                if num < maxSlicesNum:
                    destArr = np.zeros(((maxSlicesNum - num) // 2, imgsize, imgsize))
                else:
                    destArr = np.array([-1])
                for i in osw.listDir(imagepath1):
                    img = cv2.imread(osw.join(imagepath1, i), cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (imgsize, imgsize))
                    if np.all(destArr == -1):
                        destArr = np.array([img])
                    else:
                        destArr = np.vstack((destArr, np.array([img])))
                if destArr.shape[0] < maxSlicesNum:
                    destArr = np.vstack((destArr, np.zeros(((maxSlicesNum - destArr.shape[0]), imgsize, imgsize))))
                elif destArr.shape[0] > maxSlicesNum:
                    destArr = destArr[:maxSlicesNum, :, :]
                train_set.append(destArr[:clip2, :clip1, :clip1])
                train_label.append(line[2])
                train_set.append(destArr[:clip2, -1*clip1:, :clip1])
                train_label.append(line[2])

        train_set = np.array(train_set)
        train_label = np.array(train_label)
        train_arr = train_set.reshape(-1, clip2, clip1, clip1, 1)
        train_label1 = np_utils.to_categorical(train_label, num_classes=2)
        return (train_arr, train_label1)


    def trainDataGenerator(self, filepath, imgsize=128, maxSlicesNum=587, clip1=96, clip2=384):
        count0 = 0
        train_set = list()
        train_label = list()

        for f in ['train1_label.csv', 'train2_label.csv']:
            df = pd.read_csv(osw.join(filepath, f))
            count1 = 0
            for line in df.itertuples():
                # 每个csv的前150条数据不参加训练，留作验证集
                if count1< 150:
                    count1 += 1
                    continue
                print('{}_{}'.format(f, count1))
                count1 += 1
                imagepath1 = osw.join(filepath, f.split('_')[0]+'_jpg', line[1])
                num = len(osw.listDir(imagepath1))
                if num < maxSlicesNum:
                    destArr = np.zeros(((maxSlicesNum - num) // 2, imgsize, imgsize))
                else:
                    destArr = np.array([-1])
                for i in osw.listDir(imagepath1):
                    img = cv2.imread(osw.join(imagepath1, i), cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (imgsize, imgsize))
                    if np.all(destArr == -1):
                        destArr = np.array([img])
                    else:
                        destArr = np.vstack((destArr, np.array([img])))
                if destArr.shape[0] < maxSlicesNum:
                    destArr = np.vstack((destArr, np.zeros(((maxSlicesNum - destArr.shape[0]), imgsize, imgsize))))
                elif destArr.shape[0] > maxSlicesNum:
                    destArr = destArr[:maxSlicesNum, :, :]
                train_set.append(destArr[:clip2, :clip1, :clip1])
                train_label.append(line[2])
                train_set.append(destArr[:clip2, -1*clip1:, :clip1])
                train_label.append(line[2])
                train_set.append(destArr[:clip2, -1*clip1:, -1*clip1:])
                train_label.append(line[2])
                train_set.append(destArr[:clip2, :clip1:, -1*clip1:])
                train_label.append(line[2])
                train_set.append(destArr[-clip2:, :clip1, :clip1])
                train_label.append(line[2])
                train_set.append(destArr[-clip2:, -1*clip1:, :clip1])
                train_label.append(line[2])
                train_set.append(destArr[-clip2:, -1*clip1:, -1*clip1:])
                train_label.append(line[2])
                train_set.append(destArr[-clip2:, :clip1:, -1*clip1:])
                train_label.append(line[2])
                if count0 == 1:
                    #打乱数据
                    permutation = np.random.permutation(np.array(train_set).shape[0])
                    train_set = np.array(train_set)[permutation, :, :]
                    train_label = np.array(train_label)[permutation]
                    train_arr = train_set.reshape(-1, clip2, clip1, clip1, 1)
                    train_label1 = np_utils.to_categorical(train_label, num_classes=2)
                    train_set = list()
                    train_label = list()
                    count0 = 0
                    yield (train_arr, train_label1)
                else:
                    count0 += 1


    def buildModel(self):
        model = Sequential()
        model.add(Convolution3D(32, 5, strides=1, padding='same', activation='relu', input_shape=(384, 96, 96, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Convolution3D(32, 5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Convolution3D(64, 5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Convolution3D(64, 5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Convolution3D(128, 5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Convolution3D(128, 5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        # 把第二个池化层的输出扁平化为1维
        model.add(Flatten())
        # # 第一个全连接层
        # model.add(Dense(64, activation='relu'))
        # # Dropout
        # model.add(Dropout(0.5))
        # 第二个全连接层
        model.add(Dense(2, activation='softmax'))
        # 定义优化器
        adam = Adam(lr=1e-4)

        # 定义优化器，loss function，训练过程中计算准确率
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        return model


if __name__ == '__main__':
    filedir = sys.argv[1]
    m = Model()
    batchs = m.trainDataGenerator(filedir)
    # a, b = batchs.__next__()
    # print(a.shape, b.shape)
    model = m.buildModel()
    print(model.summary())
    print('--------------------加载验证集--------------------')
    valdata = m.loadValData(filedir)
    print('--------------------开始训练--------------------')
    tensorboard = TensorBoard(log_dir='result/log')
    filepath1 = "result/models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath1, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    #当评价指标不在提升时，减少学习率
    rr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    callback_lists = [tensorboard,checkpoint, rr]  # 因为callback是list型,必须转化为list
    model.fit_generator(batchs, steps_per_epoch=20, epochs=180, workers=24, use_multiprocessing=True, validation_data=valdata)
    """
    1/8 [==>...........................] - ETA: 56:53 - loss: 1.3589 - acc: 0.5000
    2/8 [======>.......................] - ETA: 44:11 - loss: 0.6795 - acc: 0.7500
    3/8 [==========>...................] - ETA: 35:05 - loss: 3.1393 - acc: 0.6667
    """
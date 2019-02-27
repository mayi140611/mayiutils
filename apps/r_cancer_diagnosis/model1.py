#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: model1.py
@time: 2019/2/26 10:30
"""
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution3D,MaxPooling3D,Flatten
from keras.optimizers import Adam
from keras.models import load_model
import os
import sys

class Model:
    def loadTrainData(self, filepath, index):
        """
        载入训练集
        :param filepath:
        :return:
        """
        paths = os.listdir('{}/batchs'.format(filepath))
        print('载入训练集：{}'.format(paths[index]))
        data = np.load(os.path.join('{}/batchs'.format(filepath), paths[index]))
        train_set = data['train_set']
        train_label = data['train_label']
        train_set = train_set.reshape(-1, 587, 64, 64, 1)
        print(train_set.shape, len(train_label))
        # 归一化
        train_set = train_set / 255.0

        # # 换one hot格式
        train_label = np_utils.to_categorical(train_label, num_classes=2)
        return train_set, train_label

    def loadValData(self, filepath, index):
        """
        载入验证集
        :param filepath:
        :return:
        """
        paths = os.listdir('{}/batchs'.format(filepath))
        print('载入验证集：{}'.format(paths[index]))
        data = np.load(os.path.join('{}/batchs'.format(filepath), paths[index]))
        val_set = data['train_set']
        val_label = data['train_label']
        val_set = val_set.reshape(-1, 587, 64, 64, 1)
        print(val_set.shape, len(val_label))
        # 归一化
        val_set = val_set / 255.0

        # # 换one hot格式
        val_label = np_utils.to_categorical(val_label, num_classes=2)
        return val_set, val_label

    def loadTestData(self, filepath, index):
        """
        载入测试集
        :param filepath:
        :param index:
        :return:
        """
        paths = os.listdir('{}/test_batchs'.format(filepath))
        print('载入测试集：{}'.format(paths[index]))
        data = np.load(os.path.join('{}/test_batchs'.format(filepath), paths[index]))
        test_set = data['test_set']
        testimgnamelist = list(data['testimgname'])
        test_set = test_set.reshape(-1, 587, 64, 64, 1)
        # 归一化
        test_set = test_set / 255
        return test_set, testimgnamelist

    def loadModel(self,filepath):
        model = load_model(filepath)
        return model

    def predict(self, model, test_set):
        r = model.predict(test_set)
        return r


    def buildModel(self):
        model = Sequential()
        model.add(Convolution3D(32, 5, strides=1, padding='same', activation='relu', input_shape=(587, 64, 64, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Convolution3D(32, 5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Convolution3D(64, 5, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Convolution3D(64, 5, strides=1, padding='same', activation='relu'))
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

    def buildModel2(self):
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
    # filepath = 'D:/Desktop/DF'
    # filepath2 = '{}/models/{}'.format(filepath, 'model_0_0_17.h5')
    filepath = sys.argv[1]
    filepath2 = '{}/models/{}'.format(filepath, sys.argv[2])
    m = Model()
    filepath = sys.argv[1]
    # model.loadData(filepath, 0, -1)
    model = m.buildModel()
    print(model.summary())
    epochs = 10
    for epoch in range(epochs):
        print('-----------------第 {} epoch begins-----------------'.format(epoch))
        for i in range(38):
            val_set, val_label = m.loadValData(filepath, i)
            for ii in range(38):
                if ii == i: continue
                train_set, train_label = m.loadTrainData(filepath, ii)
                model.fit(train_set, train_label, batch_size=20, epochs=1, validation_data=(val_set, val_label))
                model.save('models/model_{}_{}_{}.h5'.format(epoch, i, ii))
        print('-----------------第 {} epoch ends-----------------'.format(epoch))
    model.fit_generator()
    model = m.loadModel(filepath2)
    df = pd.read_csv('{}/test.csv'.format(filepath))
    #测试
    count = 0
    for i in range(21):
        test_set, name = m.loadTestData(filepath, i)
        print('开始预测')
        for ii in range(test_set.shape[0]):
            r = m.predict(model, np.array([test_set[ii]]))
            # print(r)
            # print(np.argmax(r, axis=1))
            df['ret'][df['id']==name[ii]] = np.argmax(r, axis=1)[0]
            count += 1
        print(count)
    df.to_csv('test_predict.csv', index=False)




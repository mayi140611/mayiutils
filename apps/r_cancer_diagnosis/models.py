#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: models.py
@time: 2019/3/6 13:10

定义各种模型结构
"""
import os
import pandas as pd
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import load_model


class Models:

    def __init__(self, maxSliceNum=192, imageSize=48, totalNum=7574, trainsetNum=6800, baseDir='/home/ian/datafoundation', reslutDir='result'):
        self._baseDir = baseDir
        self._totalNum = totalNum
        self._trainsetNum = trainsetNum
        self._maxSliceNum = maxSliceNum
        self.imageSize = imageSize
        self._resultDir = reslutDir

    def buildSimpleModel(self, kernel_size=3, strides=2):
        """
        目标，使得训练集快速跑出过拟合
        :return:
        """
        imagesize = self.imageSize
        model = tf.keras.Sequential()
        model.add(Conv3D(32, kernel_size, strides=strides, padding='valid', activation='relu', input_shape=(self._maxSliceNum, imagesize, imagesize, 1)))
        model.add(Conv3D(32, kernel_size, strides=strides, padding='valid', activation='relu'))
        model.add(MaxPooling3D(pool_size=2, strides=2, padding='same'))
        model.add(Conv3D(64, kernel_size, strides=strides, padding='valid', activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(MaxPooling3D(pool_size=2, strides=2, padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        optimizer = Adam(lr=1e-4)
        # optimizer= RMSprop(lr=1e-4)
        # model.add(Dense(1, activation='softmax'))
        # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model

    def buildSimpleModel1(self, kernel_size=3, strides=2):
        """
        目标，使得训练集快速跑出过拟合
        :return:
        """
        imagesize = self.imageSize
        model = tf.keras.Sequential()
        model.add(Conv3D(32, kernel_size, strides=1, padding='valid', activation='relu', input_shape=(self._maxSliceNum, imagesize, imagesize, 1)))
        model.add(Conv3D(32, kernel_size, strides=1, padding='valid', activation='relu'))
        model.add(MaxPooling3D(pool_size=2, strides=2, padding='same'))
        model.add(Conv3D(64, kernel_size, strides=1, padding='valid', activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        optimizer = Adam(lr=1e-4)
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model

    def buildVgg11Model(self, kernel_size=3, strides=2):
        """
        目标，使得训练集快速跑出过拟合
        :return:
        """
        imagesize = self.imageSize
        model = tf.keras.Sequential()
        model.add(Conv3D(32, kernel_size, strides=strides, padding='valid', activation='relu',
                         input_shape=(self._maxSliceNum, imagesize, imagesize, 1)))
        model.add(Conv3D(32, kernel_size, strides=strides, padding='valid', activation='relu'))
        model.add(MaxPooling3D(pool_size=2, strides=strides, padding='same'))
        model.add(Conv3D(64, kernel_size, strides=strides, padding='valid', activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(MaxPooling3D(pool_size=2, strides=2, padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        optimizer = Adam(lr=1e-4)
        # optimizer= RMSprop(lr=1e-4)
        # model.add(Dense(1, activation='softmax'))
        # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model

    def fit(self, model, trainDataset, valDataset, steps_per_epoch=1700, epochs=400):
        filepath1 = self._resultDir+"/models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        logdir = '{}/log'.format(self._resultDir)
        if not os.path.exists(logdir):
            print('创建目录:{}'.format(logdir))
            os.makedirs(logdir)
        modeldir = '{}/models'.format(self._resultDir)
        if not os.path.exists(modeldir):
            print('创建目录:{}'.format(modeldir))
            os.makedirs(modeldir)

        callbacks = [
            # Interrupt training if `val_loss` stops improving for over 2 epochs
            # tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            # Write TensorBoard logs to `./logs` directory
            tf.keras.callbacks.TensorBoard(log_dir='{}/log'.format(self._resultDir)),
            tf.keras.callbacks.ModelCheckpoint(filepath1, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=0, mode='auto',
                                                 cooldown=0, min_lr=0)
        ]
        print('开启训练')
        model.fit(trainDataset, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=valDataset, validation_steps=387,
                  callbacks=callbacks)
        print('train finished!')

    def predict(self, modelname, testDataset, resultFileName):
        print('加载模型{}'.format(modelname))
        model = load_model(os.path.join('result/models', modelname))
        print('开始预测')
        result = model.predict(testDataset, steps=806)
        print('预测完成')
        print(result.shape)
        r = list()
        for a in result:
            if a < 0.5:
                r.append(0)
            else:
                r.append(1)
        df = pd.read_csv(os.path.join(self._baseDir, 'test.csv'))
        df['ret'] = r
        df.to_csv(resultFileName, index=False)


if __name__ == '__main__':
    m = Models()
    m.buildSimpleModel()
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
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, Flatten, Dense, Dropout
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

    def buildSimpleModel(self, kernel_size=3, strides=2, dropout=True, pooltype='max'):
        """
        目标，使得训练集快速跑出过拟合
        :return:
        """
        imagesize = self.imageSize
        model = tf.keras.Sequential()
        model.add(Conv3D(32, kernel_size, strides=strides, padding='valid', activation='relu', input_shape=(self._maxSliceNum, imagesize, imagesize, 1)))
        model.add(Conv3D(32, kernel_size, strides=strides, padding='valid', activation='relu'))
        if pooltype == 'max':
            model.add(MaxPooling3D(pool_size=2, strides=2, padding='same'))
        elif pooltype == 'average':
            model.add(AveragePooling3D(pool_size=2, strides=2, padding='same'))
        model.add(Conv3D(64, kernel_size, strides=strides, padding='valid', activation='relu'))
        if pooltype == 'max':
            model.add(MaxPooling3D(pool_size=2, strides=2, padding='same'))
        elif pooltype == 'average':
            model.add(AveragePooling3D(pool_size=2, strides=2, padding='same'))
        model.add(Flatten())
        if dropout:
            model.add(Dropout(0.9))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        optimizer = Adam(lr=1e-4)
        # optimizer = tf.train.AdamOptimizer(1e-4)
        # optimizer= RMSprop(lr=1e-4)
        # model.add(Dense(1, activation='softmax'))
        # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        # model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model

    def buildVgg11Model(self, kernel_size=3, strides=2):
        """
        目标，使得训练集快速跑出过拟合
        :return:
        """
        pass

    def buildInceptionModel(self):
        """
        目标，尝试使用函数式的方法来搭建模型
        使用函数式 API 构建的模型具有以下特征：
            层实例可调用并返回张量。
            输入张量和输出张量用于定义 tf.keras.Model 实例。
            此模型的训练方式和 Sequential 模型一样。
        :return:
        """
        imagesize = self.imageSize
        inputs = tf.keras.Input(shape=(self._maxSliceNum, imagesize, imagesize, 1))  # Returns a placeholder tensor
        tower_1 = Conv3D(32, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
        tower_1 = Conv3D(32, kernel_size=3, strides=2, padding='same', activation='relu')(tower_1)
        tower_2 = Conv3D(32, kernel_size=1, strides=1, padding='same', activation='relu')(inputs)
        tower_2 = Conv3D(32, kernel_size=3, strides=2, padding='same', activation='relu')(tower_2)
        tower_3 = MaxPooling3D(pool_size=2, strides=2, padding='same')(inputs)
        tower_3 = Conv3D(32, kernel_size=1, strides=1, padding='same', activation='relu')(tower_3)
        # inceptionmodule1 = tf.keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
        # tower_1 = Conv3D(64, kernel_size=1, strides=1, padding='same', activation='relu')(inceptionmodule1)
        # tower_1 = Conv3D(64, kernel_size=3, strides=2, padding='same', activation='relu')(tower_1)
        # tower_2 = Conv3D(64, kernel_size=1, strides=2, padding='same', activation='relu')(inceptionmodule1)
        # tower_2 = Conv3D(64, kernel_size=5, strides=2, padding='same', activation='relu')(tower_2)
        # tower_3 = MaxPooling3D(pool_size=2, strides=2, padding='same')(inceptionmodule1)
        # tower_3 = Conv3D(64, kernel_size=1, strides=2, padding='same', activation='relu')(tower_3)
        inceptionmodule2 = tf.keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
        f = Flatten()(inceptionmodule2)
        dense = Dense(512, activation='relu')(f)
        dense = Dense(256, activation='relu')(dense)
        output = Dense(2, activation='softmax')(dense)
        model = tf.keras.Model(inputs=inputs, outputs=output)
        optimizer = Adam(lr=1e-4)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model

    def fit(self, model, trainDataset, valDataset, steps_per_epoch=3400, epochs=1000):
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
            # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=0, mode='auto',
            #                                      cooldown=0, min_lr=0)
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
        r = np.argmax(result, axis=1)
        print(r[:5])
        # r = list()
        # for a in result:
        #     if a < 0.5:
        #         r.append(0)
        #     else:
        #         r.append(1)
        df = pd.read_csv(os.path.join(self._baseDir, 'test.csv'))
        df['ret'] = r
        df.to_csv(resultFileName, index=False)


if __name__ == '__main__':
    m = Models()
    m.buildSimpleModel()
#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: bpmodel.py
@time: 2019/3/2 8:37
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os

def parse(x, y):
    print(y)
    return x, tf.keras.utils.to_categorical(y-422, num_classes=298)


class BpModel:
    def __init__(self, totalNum=50000, trainsetNum=40000, baseDir='D:/Desktop/DF/portrait'):
        self._baseDir = baseDir
        self._totalNum = totalNum
        self._trainsetNum = trainsetNum

    def prepareTrainDataset(self):
        df = pd.read_csv(os.path.join(self._baseDir, 'train_dataset.csv'))
        dfVal = df.values
        np.random.shuffle(dfVal)
        print(dfVal.shape)
        data = dfVal[: self._trainsetNum, 1:29].astype(np.float32)
        print(data.shape)
        label = dfVal[: self._trainsetNum, 29].astype(np.float32)
        label = np.array([tf.keras.utils.to_categorical(y-422, num_classes=298) for y in label])
        print(label.shape)
        print(label[:5])
        dataset = tf.data.Dataset.from_tensor_slices((data, label))
        dataset = dataset.shuffle(self._trainsetNum).repeat(1000).batch(100)
        return dataset

    def prepareValDataset(self):
        df = pd.read_csv(os.path.join(self._baseDir, 'train_dataset.csv'))
        dfVal = df.values
        np.random.shuffle(dfVal)
        print(dfVal.shape)
        data = dfVal[self._trainsetNum:, 1:29].astype(np.float32)
        print(data.shape)
        label = dfVal[self._trainsetNum:, 29].astype(np.float32)
        label = np.array([tf.keras.utils.to_categorical(y-422, num_classes=298) for y in label])
        print(label.shape)
        dataset = tf.data.Dataset.from_tensor_slices((data, label))
        dataset = dataset.repeat(100).batch(100)
        return dataset

    def prepareTestDataset(self):
        df = pd.read_csv(os.path.join(self._baseDir, 'test_dataset.csv'))
        dfVal = df.values
        np.random.shuffle(dfVal)
        print(dfVal.shape)
        data = dfVal[:, 1:29].astype(np.float32)
        print(data.shape)
        label = np.zeros((dfVal.shape[0])).astype(np.float32)
        label = np.array([tf.keras.utils.to_categorical(0, num_classes=298) for y in label])
        print(label.shape)
        dataset = tf.data.Dataset.from_tensor_slices((data, label))
        dataset = dataset.batch(1000)
        return dataset

    def buildModel(self):
        model = tf.keras.Sequential()
        model.add(Dense(500, activation='sigmoid', input_shape=(28,)))
        model.add(Dense(298, activation='softmax'))
        adam = Adam(lr=1e-4)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def loadModel(self, filename):
        return load_model(os.path.join(self._baseDir, 'result/models', filename))


if __name__ == '__main__':
    mode = '2'
    m = BpModel()
    testDataset = m.prepareTestDataset()
    model = m.loadModel('weights-improvement-99-0.22.hdf5')
    result = model.predict(testDataset, steps=50)
    r = np.argmax(result, axis=1)
    print(r[:5]+422)#[237 209 221 120 232]
    df = pd.read_csv(os.path.join(m._baseDir, 'submit_example.csv'))
    df['score'] = r+422
    df.to_csv(os.path.join(m._baseDir, 'test1.csv'), index=False)

    if mode == '1':
        model = m.buildModel()
        print(model.summary())
        trainDataset = m.prepareTrainDataset()
        valDataset = m.prepareValDataset()
        filepath1 = os.path.join(m._baseDir, "result/models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
        callbacks = [
            # Interrupt training if `val_loss` stops improving for over 2 epochs
            # tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            # Write TensorBoard logs to `./logs` directory
            tf.keras.callbacks.TensorBoard(log_dir=os.path.join(m._baseDir, "result/log")),
            tf.keras.callbacks.ModelCheckpoint(filepath1, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', cooldown=0, min_lr=0)
        ]
        print('开启训练')
        model.fit(trainDataset, steps_per_epoch=4000, epochs=100, validation_data=valDataset, validation_steps=100, callbacks=callbacks)

#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: tfkeras_wrapper.py
@time: 2019/2/28 9:02
tensorflow==1.13.1
tf.keras.__version__==2.2.4-tf
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

# class TfModel:


if __name__ == '__main__':
    mode = 2
    if mode == 2:
        print(tf.keras.utils.to_categorical(1, 5))#[0. 1. 0. 0. 0.]
        print(tf.keras.utils.to_categorical([1], 5))#[[0. 1. 0. 0. 0.]]
        print(tf.keras.utils.to_categorical([1, 2], 5))
        """
        [[0. 1. 0. 0. 0.]
        [0. 0. 1. 0. 0.]]
        """
        print(tf.keras.utils.to_categorical([[1]], 5))#[[0. 1. 0. 0. 0.]]
        print(tf.keras.utils.to_categorical([[1, 2]], 5))
        """
        [[[0. 1. 0. 0. 0.]
        [0. 0. 1. 0. 0.]]]
        """
        print(tf.keras.utils.to_categorical([[1], [2]], 5))
        """
        和tf.keras.utils.to_categorical([1, 2], 5)效果相同
        [[0. 1. 0. 0. 0.]
        [0. 0. 1. 0. 0.]]
        """
    if mode == 1:
        print(tf.VERSION)
        print(tf.keras.__version__)
        model = tf.keras.Sequential()
        model.add(Conv3D(32, 5, strides=1, padding='same', activation='relu', input_shape=(384, 96, 96, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Conv3D(32, 5, strides=1, padding='same', activation='relu', input_shape=(384, 96, 96, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Conv3D(64, 5, strides=1, padding='same', activation='relu', input_shape=(384, 96, 96, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Conv3D(64, 5, strides=1, padding='same', activation='relu', input_shape=(384, 96, 96, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))
        adam = Adam(lr=1e-4)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())


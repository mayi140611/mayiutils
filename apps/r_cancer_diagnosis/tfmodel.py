#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: model3.py
@time: 2019/2/27 14:48
"""
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

import os
def parse4(x, y):
    print(x, y)
    return x,y
def parse(dirname, label):
    # dir1 = 'D:/Desktop/DF/train1_jpg'
    dir1 = '/home/ian/datafoundation/train1_jpg'
    imgs = [os.path.join(dir1, dirname.decode(), i) for i in os.listdir(os.path.join(dir1, dirname.decode()))]
    return ','.join(imgs), label


def parse2(dirname, label):
    """
    报错：    if x.shape is not None and len(x.shape) == 1:
  File "/home/ian/installed/anaconda3/lib/python3.6/site-packages/tensorflow/python/framework/tensor_shape.py", line 579, in __len__
    raise ValueError("Cannot take the length of Shape with unknown rank.")
ValueError: Cannot take the length of Shape with unknown rank.
    :param dirname:
    :param label:
    :return:
    """
    # arr = np.arange(9)
    # print(dirname.decode())
    imglist = dirname.decode().split(',')
    # print(imglist)
    count = 0
    imgsize = 64
    for filename in imglist:
        # filename = 'D:/Desktop/DF/train1_jpg/{}'.format(i)
        # print(filename)
        # image_string = tf.read_file(filename)
        # image_decoded = tf.image.decode_jpeg(image_string)
        # image_resized = tf.image.resize_images(image_decoded, [64, 64])
        arr1 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        arr1 = cv2.resize(arr1, (imgsize, imgsize))
        if count == 0:
            arr = np.array([arr1])
            count += 1
        else:
            arr = np.vstack((arr, np.array([arr1])))

    print(arr.shape)
    if arr.shape[0] < 384:
        arr = np.vstack((arr, np.zeros((384-arr.shape[0], imgsize, imgsize))))
    else:
        arr = arr[:384, :, :]
    arr = arr.reshape(-1, imgsize, imgsize, 1)
    print(arr.shape)
    return arr, np.array([label])

def parse3(dirname, label):
    imglist = dirname.decode().split(',')
    count = 0
    imgsize = 64
    for filename in imglist:
        # filename = 'D:/Desktop/DF/train1_jpg/{}'.format(i)
        print(filename)
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        arr1 = tf.image.resize_images(image_decoded, [imgsize, imgsize])

        if count == 0:
            arr = np.array([arr1])
            count += 1
        else:
            arr = np.vstack((arr, np.array([arr1])))

    print(arr.shape)
    if arr.shape[0] < 384:
        arr = np.vstack((arr, np.zeros((384-arr.shape[0], imgsize, imgsize))))
    else:
        arr = arr[:384, :, :]
    arr = arr.reshape(-1, imgsize, imgsize, 1)
    print(arr.shape)
    return arr, label
class Model:

    def buildModel(self):
        model = tf.keras.Sequential()
        model.add(Conv3D(32, 5, strides=1, padding='same', activation='relu', input_shape=(384, 64, 64, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Conv3D(32, 5, strides=1, padding='same', activation='relu', input_shape=(384, 64, 64, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Conv3D(64, 5, strides=1, padding='same', activation='relu', input_shape=(384, 64, 64, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Conv3D(64, 5, strides=1, padding='same', activation='relu', input_shape=(384, 64, 64, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same'))
        model.add(Flatten())
        model.add(Dense(1, activation='softmax'))
        adam = Adam(lr=1e-4)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def prepareDataset(self):
        filenames = ['/home/ian/datafoundation/train1_label.csv']
        record_defaults = [['aa'], [0]]
        dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True)
        dataset1 = dataset.map(
            lambda dirname, label: tuple(tf.py_func(parse, [dirname, label], [tf.string, label.dtype])))
        dataset2 = dataset1.map(
            lambda dcmlist, label: tuple(tf.py_func(parse2, [dcmlist, label], [tf.double, label.dtype])))
        dataset2 = dataset2.map(parse4)
        dataset = dataset2.repeat(10)
        dataset = dataset.batch(32)
        return dataset


if __name__ == '__main__':
    # filedir = sys.argv[1]
    m = Model()

    model = m.buildModel()
    print(model.summary())
    dataset = m.prepareDataset()
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    sess = tf.Session()
    sess.run(iterator.initializer)
    sess.run(next_element)
    # print(sess.run(next_element))
    # tf.Tensor.set_shape()
    # model.fit(dataset, epochs=10, steps_per_epoch=30, validation_data=dataset)
    # print('--------------------加载验证集--------------------')
    # valdata = m.loadValData(filedir)
    # print('--------------------开始训练--------------------')
    # tensorboard = TensorBoard(log_dir='result/log')
    # filepath1 = "result/models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath1, monitor='val_acc', verbose=1, save_best_only=True,
    #                              mode='max')
    #当评价指标不在提升时，减少学习率
    # rr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    # callback_lists = [tensorboard,checkpoint, rr]  # 因为callback是list型,必须转化为list

    # history = model.fit_generator(batchs, steps_per_epoch=20, epochs=180, workers=24, use_multiprocessing=True,
    #                               validation_data=valdata)

    # history = model.fit_generator(dg, steps_per_epoch=20, epochs=180, validation_data=valdata, callbacks=callback_lists, workers=24, use_multiprocessing=True)
    model.save('m190227.h5')
    # pkw.dump2File(history, 'history190227.pkl')
    print('--------------------训练完成--------------------')
    """
    1/8 [==>...........................] - ETA: 56:53 - loss: 1.3589 - acc: 0.5000
    2/8 [======>.......................] - ETA: 44:11 - loss: 0.6795 - acc: 0.7500
    3/8 [==========>...................] - ETA: 35:05 - loss: 3.1393 - acc: 0.6667
    """
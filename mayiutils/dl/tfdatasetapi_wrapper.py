#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: tfdatasetapi_wrapper.py
@time: 2019/2/28 10:05

tensorflow==1.13.1
"""
import tensorflow as tf
from tensorflow.data import Dataset
import os
import numpy as np
import cv2

def parse(dirname, label):
    # dir1 = 'D:/Desktop/DF/train1_jpg'
    dir1 = '/home/ian/datafoundation/train1_jpg'
    imgs = [os.path.join(dir1, dirname.decode(), i) for i in os.listdir(os.path.join(dir1, dirname.decode()))]
    return ','.join(imgs), label

def parse2(dirname, label):
    # arr = np.arange(9)
    # print(dirname.decode())
    imglist = dirname.decode().split(',')
    # print(imglist)
    count = 0
    imgsize = 64
    for filename in imglist:
        # filename = 'D:/Desktop/DF/train1_jpg/{}'.format(i)
        print(filename)
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
    return arr, label

def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized


if __name__=='__main__':
    select = 0
    arr = np.load('D:/Desktop/DF/filesamples.npy.npz')
    # np.cast()
    print(arr['samplelist'].shape )
    # print(np.array(arr[:, 0]).shape)
    # print(arr[:, 1, np.newaxis])
    dataset1 = Dataset.from_tensor_slices((arr['samplelist'], arr['labellist']))
    print(dataset1.output_types)  # ==> "tf.float32"
    print(dataset1.output_shapes)  # ==> "(10,)"
    if select == 1:
        # dataset1 = Dataset.from_tensor_slices(tf.random_uniform([4, 10, 20]))
        # filenames = ['D:/Desktop/DF/train1_label.csv', 'D:/Desktop/DF/train2_label.csv']
        filenames = ['/home/ian/datafoundation/train1_label.csv']
        record_defaults = [['aa'], [0]]
        dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True)
        dataset1 = dataset.map(lambda dirname, label: tuple(tf.py_func(parse, [dirname, label], [tf.string, label.dtype])))
        dataset2 = dataset1.map(lambda dcmlist, label: tuple(tf.py_func(parse2, [dcmlist, label], [tf.double, label.dtype])))
        # dataset1 = Dataset.from_tensor_slices(filenames)
        # arr = np.load('D:/Desktop/DF/filesamples.npy')
        # dataset1 = Dataset.from_tensor_slices((arr[:, 0], arr[:, 1]))
        # dataset2 = dataset1.map(_parse_function)
        # print(dataset1.output_types)  # ==> "tf.float32"
        # print(dataset1.output_shapes)  # ==> "(10,)"
        iterator = dataset2.make_initializable_iterator()
        next_element = iterator.get_next()
        sess = tf.Session()
        sess.run(iterator.initializer)
        print(sess.run(next_element))
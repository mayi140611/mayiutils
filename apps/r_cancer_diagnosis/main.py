#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: main.py
@time: 2019/3/6 11:02
"""
# from apps.r_cancer_diagnosis.models import Models
# from apps.r_cancer_diagnosis.data_prepare import DataPrepare
from models import Models
from data_prepare import DataPrepare
import tensorflow as tf
import sys

if __name__ == "__main__":
    # tf.enable_eager_execution()
    # print(tf.executing_eagerly())
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = '0'
    m = Models()
    if mode == '0':
        print('查看模型结构')
        # m = Models(imageSize=256)
        # model = m.buildSimpleModel(6, 4)
        m = Models(imageSize=48)
        # model = m.buildSimpleModel(6, 4)
        model = m.buildInceptionModel()
    elif mode == 'train':
        print('训练模式')
        imagesize = int(sys.argv[2])
        print('imagesize:', imagesize)
        kernel_size = int(sys.argv[3])
        print('kernel_size: ', kernel_size)
        strides = int(sys.argv[4])
        print('strides: ', strides)
        dropout = sys.argv[5]
        if dropout == 'true':
            dropout = True
        if dropout == 'false':
            dropout = False
        print('dropout: ', dropout)
        pooltype = sys.argv[6]
        print('pooltype: ', pooltype)
        dp = DataPrepare(imageSize=imagesize)
        m = Models(imageSize=imagesize)
        imagePatharr, labelarr = dp.prepareTrainandValImagePath()
        trainDataset = dp.prepareTrainDataSet(imagePatharr, labelarr)
        valDataset = dp.prepareValDataset(imagePatharr, labelarr)
        model = m.buildSimpleModel(kernel_size=kernel_size, strides=strides, dropout=dropout, pooltype=pooltype)
        m.fit(model, trainDataset, valDataset)
    elif mode == 'train_inception':
        print('训练模式inception')
        imagesize = int(sys.argv[2])
        print('imagesize:', imagesize)
        dp = DataPrepare(imageSize=imagesize)
        m = Models(imageSize=imagesize)
        imagePatharr, labelarr = dp.prepareTrainandValImagePath()
        trainDataset = dp.prepareTrainDataSet(imagePatharr, labelarr)
        valDataset = dp.prepareValDataset(imagePatharr, labelarr)
        model = m.buildInceptionModel()
        m.fit(model, trainDataset, valDataset)
    elif mode == 'test':
        print('预测模式')
        imagesize = int(sys.argv[2])
        print('imagesize:', imagesize)
        modelname = sys.argv[3]
        dp = DataPrepare(imageSize=imagesize)
        testPatharr = dp.prepareTestImagePath()
        testDataset = dp.prepareTestDataset(testPatharr)
        m.predict(modelname, testDataset, 'result.csv')









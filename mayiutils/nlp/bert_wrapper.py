#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: bert_wrapper.py
@time: 2019-04-26 14:23





"""
from mayiutils.nlp.bert.extract_feature import BertVector
from mayiutils.nlp.bert.similarity import BertSim
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    """
    https://blog.csdn.net/u012526436/article/details/84637834
    https://github.com/terrifyzhao/bert-utils
    https://github.com/google-research/bert
    """
    mode = 2
    if mode == 2:
        """
        文本分类
        文本分类需要做fine tune，
        首先把数据准备好存放在data目录下，训练集的名字必须为train.csv，验证集的名字必须为dev.csv，测试集的名字必须为test.csv， 
        必须先调用set_mode方法，可参考similarity.py的main方法，
        """
        sim = BertSim()
        # sim.set_mode(tf.estimator.ModeKeys.TRAIN)
        # sim.train()
        # sim.set_mode(tf.estimator.ModeKeys.EVAL)
        # sim.eval()
        sim.set_mode(tf.estimator.ModeKeys.PREDICT)
        while True:
            sentence1 = input('sentence1: ')
            sentence2 = input('sentence2: ')
            predict = sim.predict(sentence1, sentence2)
            print(predict)
            print(f'similarity：{predict[0][1]}')
    if mode == 1:
        """
        句向量生成
        
        生成句向量不需要做fine tune，使用预先训练好的模型即可，可参考extract_feature.py的main方法，注意参数必须是一个list。

        首次生成句向量时需要加载graph，并在output_dir路径下生成一个新的graph文件，因此速度比较慢，再次调用速度会很快
        """
        bv = BertVector()
        vec = bv.encode(['今天天气不错',
                          '今天天气很好',
                          '今天天气很差'])

        print(type(vec))#<class 'numpy.ndarray'>
        print(vec.shape)#(3, 768)
        # print(vec1)
        vec1 = vec[0]
        vec2 = vec[1]
        vec3 = vec[2]
        # 余弦相似度
        print(vec1.dot(vec2)/(np.sqrt(vec1.dot(vec1))*np.sqrt(vec2.dot(vec2))))#0.9787841
        print(vec1.dot(vec3)/(np.sqrt(vec1.dot(vec1))*np.sqrt(vec3.dot(vec3))))#0.95218825
        print(vec2.dot(vec3)/(np.sqrt(vec2.dot(vec2))*np.sqrt(vec3.dot(vec3))))#0.97410476

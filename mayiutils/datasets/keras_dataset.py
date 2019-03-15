#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: keras_dataset.py
@time: 2019/3/15 10:00
"""
from keras.datasets import imdb


if __name__ == "__main__":
    """
    IMDB数据集

    IMDB数据集有5万条来自网络电影数据库的评论；
    其中2万5千条用来训练，2万5千条用来测试，每个部分正负评论各占50%.
    """
    max_features = 2000
    #num_words: max number of words to include
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)










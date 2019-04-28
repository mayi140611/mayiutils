#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: kmeans1.py
@time: 2019/3/19 9:32
"""
from mayiutils.db.pymongo_wrapper import PyMongoWrapper as pmw
from mayiutils.pickle_wrapper import PickleWrapper as picklew
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import jieba
import pandas as pd


if __name__ == '__main__':
    mode = 2
    if mode == 2:
        """
        kmeans聚类
        """
        X = picklew.loadFromFile('result.pkl')
        illness_names = picklew.loadFromFile('illness_names.pkl')
        model = KMeans(n_clusters=14, random_state=0)
        y_pred = model.fit_predict(X)
        df = pd.DataFrame({'name': illness_names, 'class': y_pred})
        df.to_csv('result.csv', index=False, encoding='gbk')
    if mode == 1:
        """
        文档特征提取
        """
        pmw = pmw('h1')
        table = pmw.getCollection('jiankang39', 'diseases')

        illness_names = []
        ill_abstracts = []
        for i in pmw.findAll(table, fieldlist=['疾病名称', '简介']):
            if '疾病名称' in i and '简介' in i:
                illness_names.append(i['疾病名称'])
                ill_abstracts.append(' '.join(jieba.lcut(i['简介'])))
        picklew.dump2File(illness_names, 'illness_names.pkl')
        picklew.dump2File(ill_abstracts, 'ill_abstracts.pkl')
        f = open('../../mayiutils/nlp/stopwords_zh.dic', encoding='utf8')
        stopwords = [i.strip() for i in f.readlines()]
        tfidf = TfidfVectorizer(stop_words=stopwords)
        result = tfidf.fit_transform(ill_abstracts).toarray()
        print(result.shape)#(7450, 29085)
        picklew.dump2File(result, 'result.pkl')

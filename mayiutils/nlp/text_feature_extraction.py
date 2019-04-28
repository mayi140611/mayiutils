#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: text_classification.py
@time: 2019/3/19 9:05
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import jieba
import re
from sklearn.metrics.pairwise import pairwise_distances, cosine_distances, cosine_similarity, paired_cosine_distances
import seaborn as sb
from matplotlib import  pyplot as plt


if __name__ == "__main__":
    mode = 3
    if mode == 3:
        """
        tfidf实例
        """
        df = pd.read_csv('/Users/luoyonggui/Documents/work/dataset/1/icd10_leapstack.csv')
        def t(x):
            x = re.sub(r'[a-zA-Zαβγδ]+', 'alphabet', x)
            x = re.sub(r'[0-9]+', 'num', x)#把数字替换为num
            return ' '.join(jieba.lcut(x))
        df['diag_name_'] = df['diag_name'].apply(t)
        # df['diag_name_'] = df['diag_name'].apply(lambda x: ' '.join(jieba.lcut(x)))
        x_test = df['diag_name_']
        with open('stopwords_zh.dic', encoding='utf8') as f:
            stopwords = [s.strip() for s in f.readlines()]
        # print(stopwords)
        tfidf = TfidfVectorizer(stop_words=stopwords, token_pattern=r"(?u)\b\w+\b")
        tfidf.fit(x_test)
        # print(tfidf.vocabulary_)
        # print(tfidf.get_feature_names())
        # print(tfidf.stop_words_)
        r = tfidf.transform(x_test).toarray()
        # np.save('tfidf_features.npy', r)
        # print(r.shape)
        # # sim = pairwise_distances(r[:10], metric='cosine')
        # sim = cosine_similarity(r)
        # sb.heatmap(sim)
        # plt.show()
    if mode == 2:
        """
        tf-idf
        fit：Learn vocabulary and idf from training set.
        transform: Transform documents to document-term matrix.
        """
        X_test = ['没有 你 的 地方 都是 他乡 没有 。 . , : ',
                  '没有 你 的 旅行 都是 流浪']
        with open('stopwords_zh.dic', encoding='utf8') as f:
            stopwords = [s.strip() for s in f.readlines()]
        print(stopwords)
        # tfidf = TfidfVectorizer(stop_words=stopwords, token_pattern=r"(?u)\b\w+\b")
        tfidf = TfidfVectorizer(max_df=0.9, token_pattern=r"(?u)\b\w+\b")
        """
        关于设置停用词的两种方式：
            1、直接设置stop_words
            2、设置max_df, df(document freq)，
            感觉第二种方式比较灵活一些，不同的场景下停用词应该有所差别才对！实际使用时，对于短文本效果并不好
        token_pattern : string (default=r"(?u)\b\w\w+\b")
            默认选择word长度大于1的word，word长度等于1的会被当做停用词忽略，这个是不合理的
            Regular expression denoting what constitutes a "token", only used
            if ``analyzer == 'word'``. The default regexp selects tokens of 2
            or more alphanumeric characters (punctuation is completely ignored
            and always treated as a token separator).
        norm : 'l1', 'l2' or None, optional (default='l2')
            把输出向量的长度归一化
            Each output row will have unit norm, either:
            * 'l2': Sum of squares of vector elements is 1. The cosine
            similarity between two vectors is their dot product when l2 norm has
            been applied.
            * 'l1': Sum of absolute values of vector elements is 1.
            See :func:`preprocessing.normalize`
        stop_words : string {'english'}, list, or None (default=None)
            If a string, it is passed to _check_stop_list and the appropriate stop
            list is returned. 'english' is currently the only supported string
            value.
            There are several known issues with 'english' and you should
            consider an alternative (see :ref:`stop_words`).
    
            If a list, that list is assumed to contain stop words, all of which
            will be removed from the resulting tokens.
            Only applies if ``analyzer == 'word'``.
    
            If None, no stop words will be used. max_df can be set to a value
            in the range [0.7, 1.0) to automatically detect and filter stop
            words based on intra corpus document frequency of terms.
        max_df : float in range [0.0, 1.0] or int (default=1.0)
            When building the vocabulary ignore terms that have a document
            frequency strictly higher than the given threshold (corpus-specific
            stop words).
            If float, the parameter represents a proportion of documents, integer
            absolute counts.
            This parameter is ignored if vocabulary is not None.
        """
        tfidf.fit(X_test)
        print(tfidf.vocabulary_)
        """
        {'没有': 4, '你': 1, '的': 6, '地方': 2, '他乡': 0, '旅行': 3, '流浪': 5}
        """
        print(tfidf.stop_words_)
        print(tfidf.get_feature_names())#['他乡', '地方', '旅行', '没有', '流浪']
        # print(tfidf.transform(X_test))
        """
          (0, 3)	0.37131279241563214
          (0, 1)	0.3143436037921839
          (0, 0)	0.3143436037921839
          (1, 4)	0.36015410466295694
          (1, 3)	0.27969179067408617
          (1, 2)	0.36015410466295694
          (2, 4)	0.36015410466295694
          (2, 3)	0.27969179067408617
          (2, 2)	0.36015410466295694
        """
        # print(tfidf.transform(X_test).toarray())
        """
        array([[0.6316672 , 0.6316672 , 0.        , 0.44943642, 0.        ],
       [0.        , 0.        , 0.6316672 , 0.44943642, 0.6316672 ]])
        """
        # print(tfidf.transform(X_test).todense().tolist())
        """
        [[0.6316672017376245, 0.6316672017376245, 0.0, 0.4494364165239821, 0.0],
        [0.0, 0.0, 0.6316672017376245, 0.4494364165239821, 0.6316672017376245]]
        """
        # print(tfidf.transform(X_test).todense())
        """
        matrix([[0.6316672 , 0.6316672 , 0.        , 0.44943642, 0.        ],
        [0.        , 0.        , 0.6316672 , 0.44943642, 0.6316672 ]])
        
        """
        print(tfidf.transform(['a b 你 他乡']).toarray())
    if mode == 1:
        """
        将doc转化成了由词频表示的特征
        """
        X_test = ['I sed about sed the lack',
                  'of any Actually']
        count_vec = CountVectorizer()
        count_vec.fit(X_test)
        """
        CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
            tokenizer=None, vocabulary=None)
        """
        print('vocabulary list:', count_vec.vocabulary_)# {'sed': 5, 'about': 0, 'the': 6, 'lack': 3, 'of': 4, 'any': 2, 'actually': 1}
        print(count_vec.get_feature_names())

        count_vec.transform(X_test)

        print(count_vec.transform(X_test))
        """
        # (index1,index2) count中：
        # index1表示为第几个句子或者文档，
        # index2为所有语料库中的单词组成的词典的序号。
        # count为在这个文档中这个单词出现的次数。
        注意：这样统计时丢失了word在text中的位置信息
          (0, 0)	1
          (0, 3)	1
          (0, 5)	2
          (0, 6)	1
          (1, 1)	1
          (1, 2)	1
          (1, 4)	1   
        """
        print(count_vec.transform(X_test).toarray())
        """
        doc-token矩阵：每一行表示一个文档，每一列表示相应编号的token。值为token在doc中出现的频数。
        这一步已经将doc转化成了由词频表示的特征
        [[1 0 0 1 0 2 1]
            [0 1 1 0 1 0 0]]
        """






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


if __name__ == "__main__":
    mode = 2
    if mode == 2:
        """
        tf-idf
        fit：Learn vocabulary and idf from training set.
        transform: Transform documents to document-term matrix.
        """
        X_test = ['没有 你 的 地方 都是 他乡 没有',
                  '没有 你 的 旅行 都是 流浪',
                  '没有 你 的 旅行 都是 流浪']
        stopwords = ['都是']
        # norm:归一化，值为l2时，只把向量的长度=1，即sqrt(x1^2+x^2...)=1,值为l1，即abs(x1)+abs(x2)...=1
        tfidf = TfidfVectorizer(stop_words=stopwords)
        tfidf.fit(X_test)
        print(tfidf.vocabulary_)
        """
        {'没有': 3, '地方': 1, '他乡': 0, '旅行': 2, '流浪': 4}
        """
        print(tfidf.get_feature_names())#['他乡', '地方', '旅行', '没有', '流浪']
        print(tfidf.transform(X_test))
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
        print(tfidf.transform(X_test).toarray())
        """
        array([[0.6316672 , 0.6316672 , 0.        , 0.44943642, 0.        ],
       [0.        , 0.        , 0.6316672 , 0.44943642, 0.6316672 ]])
        """
        print(tfidf.transform(X_test).todense().tolist())
        """
        [[0.6316672017376245, 0.6316672017376245, 0.0, 0.4494364165239821, 0.0],
        [0.0, 0.0, 0.6316672017376245, 0.4494364165239821, 0.6316672017376245]]
        """
        print(tfidf.transform(X_test).todense())
        """
        matrix([[0.6316672 , 0.6316672 , 0.        , 0.44943642, 0.        ],
        [0.        , 0.        , 0.6316672 , 0.44943642, 0.6316672 ]])
        """
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






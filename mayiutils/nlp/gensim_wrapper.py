#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: gensim_wrapper.py
@time: 2019/3/14 21:37

gensim – Topic Modelling in Python
pip install -U gensim
https://radimrehurek.com/gensim/tutorial.html
https://github.com/RaRe-Technologies/gensim
"""
import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.similarities import WmdSimilarity
from collections import defaultdict
from gensim import corpora
from gensim import models
import logging


class GensimWrapper:
    """

    """
    @classmethod
    def trainWord2vec(cls, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, sample=1e-3, seed=1, workers=3):
        """
        训练word2vec
        :param sentences:
            The sentences iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
        :param size: Dimensionality of the word vectors.
        :param alpha:
        :param window: 窗口的长度
            Maximum distance between the current and predicted word within a sentence.
        :param min_count: Ignores all words with total frequency lower than this.
        :param max_vocab_size:
        :param sample:
        :param seed:
        :param workers:
            Use these many worker threads to train the model (=faster training with multicore machines)
        :return:
        """
        return Word2Vec(sentences, size, alpha, window, min_count, max_vocab_size, sample, seed, workers)
    @classmethod
    def loadWord2vecModel(cls, path):
        """

        :param path:
        :return:
        """
        return Word2Vec.load(path)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    mode = 1
    sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
    if mode == 2:
        """
        fasttext
        """
        sentences = [["你", "是", "谁"], ["我", "是", "中国人"]]
        model = FastText(sentences)
        print(model['你'])# 词向量获得的方式
        print(model.wv['你'])# 词向量获得的方式
        print(model.wv.most_similar('你'))
    if mode == 1:
        """
        https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/gensim Quick Start.ipynb
        
        原始语料经过分词，去除停用词，字典编号，
        这样原始语料就可以转化为vector
        使用作为model的输入
        """
        raw_corpus = ["Human machine interface for lab abc computer applications",
                      "A survey of user opinion of computer system response time",
                      "The EPS user interface management system",
                      "System and human system engineering testing of EPS",
                      "Relation of user perceived response time to error measurement",
                      "The generation of random binary unordered trees",
                      "The intersection graph of paths in trees",
                      "Graph minors IV Widths of trees and well quasi ordering",
                      "Graph minors A survey"]
        # Create a set of frequent words
        stoplist = set('for a of the and to in'.split(' '))
        # Lowercase each document, split it by white space and filter out stopwords
        texts = [[word for word in document.lower().split() if word not in stoplist]
                 for document in raw_corpus]
        # Count word frequencies
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1

        # Only keep words that appear more than once
        processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
        print(processed_corpus)
        dictionary = corpora.Dictionary(processed_corpus)
        print(dictionary)#Dictionary(12 unique tokens: ['computer', 'human', 'interface', 'response', 'survey']...)
        print(dictionary.token2id)
        """
        {'computer': 0, 'human': 1, 'interface': 2, 'response': 3, 'survey': 4, 'system': 5, 'time': 6, 'user': 7, 'eps': 8, 'trees': 9, 'graph': 10, 'minors': 11}
        """
        new_doc = "Human computer interaction"
        new_vec = dictionary.doc2bow(new_doc.lower().split())
        print(new_vec)#[(0, 1), (1, 1)]
        """
        The first entry in each tuple corresponds to the ID of the token in the dictionary, the second corresponds to the count of this token.
Note that "interaction" did not occur in the original corpus and so it was not included in the vectorization. 
        """
        bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
        print(bow_corpus)
        # train the model
        tfidf = models.TfidfModel(bow_corpus)
        # transform the "system minors" string
        print(tfidf[dictionary.doc2bow("system minors".lower().split())])#[(5, 0.5898341626740045), (11, 0.8075244024440723)]


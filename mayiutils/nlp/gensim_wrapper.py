#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: gensim_wrapper.py
@time: 2019/3/14 21:37
"""
import gensim
from gensim.models import Word2Vec

class GensimWrapper:
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
    # gensim.models.Word2Vec()
    sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
    model = Word2Vec(sentences, min_count=1)
    r = model.wv.most_similar('cat')
    print(r)
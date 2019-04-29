#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: nltk_wrapper.py
@time: 2019/3/19 10:56
"""
import nltk
# from nltk.book import *
from nltk.corpus import reuters
from nltk.util import trigrams
from nltk.util import bigrams
from nltk.text import TextCollection


class NltkWrapper:
    """
    nltk中存储了很多的语料，提供了很多现成的api，方便用户进行nlp研究。主要是英文语料。
    nltk中的语料库介绍：https://www.jianshu.com/writer#/notebooks/19086358/notes/21738779/preview
    nltk中有以下几个概念
    raw：原始文本，可以看作一个大的string
    text
    words：word list。[w1,w2,w3...]
    sents：句子list。以'. '和换行符'\n'等作为分隔符的词链表。如下形式：
        [[w1,w2...],[w3,w4,...],...]
    相互转换
    raw转text: 需要先转成words list, 如 nltk.Text(raw.split())
    raw转words list：raw.split()
    raw转sents list: ?
    text转raw：''.join(list(text))
    text转words list: list(text)
    text转sents：？
    words转raw：''.join(words)
    words转text: nltk.Text(words)
    words转sents: ?
    sents转words: [w for s in sents for w in s]
    sents转raw
    sents转text
    """
    def __init__(self):
        pass


if __name__ == '__main__':
    mode = 1
    # nltk.download()
    # print(reuters.fileids())
    if mode == 1:
        """
        nltk实现tfidf 有问题！！！
        """
        corpus = TextCollection(['this is sent one',
                                 'this is sent two',
                                 'this is sent three',
                                 ])
        print(corpus.vocab())
        print(list(corpus.vocab()))
        #直接就能算出tfidf
        print(corpus.tf_idf('this', 'this is sent four'))
        print(corpus.tf_idf('s', 'this is sent four'))
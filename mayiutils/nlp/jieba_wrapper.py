#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: jieba_wrapper.py
@time: 2019/2/18 17:25
"""
import jieba


class JieBaWrapper(object):
    def __init__(self):
        pass

    @classmethod
    def cut(cls, sentence, cut_all=False, HMM=True):
        '''
        The main function that segments an entire sentence that contains
        Chinese characters into seperated words.
        :return: generator
        '''
        return jieba.cut(sentence, cut_all=cut_all, HMM=HMM)

    @classmethod
    def lcut(cls,sentence, cut_all=False, HMM=True):
        '''
        The main function that segments an entire sentence that contains
        Chinese characters into seperated words.
        :return: list
        '''
        return jieba.lcut(sentence, cut_all=cut_all, HMM=HMM)

    @classmethod
    def cut_for_search(cls,sentence, HMM=True):
        '''
        Finer segmentation for search engines.
        分词粒度较细，适合搜索引擎构建倒排索引
        :return: generator
        '''
        return jieba.cut_for_search(sentence, HMM=HMM)

    @classmethod
    def lcut_for_search(cls,sentence, HMM=True):
        '''
        Finer segmentation for search engines.
        分词粒度较细，适合搜索引擎构建倒排索引
        :return: list
        '''
        return jieba.lcut_for_search(sentence, HMM=HMM)
    '''
    关于修改jieba的字典，有四个方法
    使用 add_word(word, freq=None, tag=None) 和 del_word(word) 可在程序中动态修改词典。
    使用 suggest_freq(segment, tune=True) 可调节单个词语的词频，使其能（或不能）被分出来。
    '''
    @classmethod
    def loadUserDict(cls, filename):
        '''
        开发者可以指定自己自定义的词典，以便包含 jieba 词库里没有的词。
        虽然 jieba 有新词识别能力，但是自行添加新词可以保证更高的正确率
        词典格式和 dict.txt 一样，一个词占一行；
        每一行分三部分：词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒。
        file_name 若为路径或二进制方式打开的文件，则文件必须为 UTF-8 编码。
        词频省略时使用自动计算的能保证分出该词的词频。
        '''
        return jieba.load_userdict(filename)

    @classmethod
    def addWord(cls, word, freq=None, tag=None):
        '''
        Add a word to dictionary.

        freq and tag can be omitted, freq defaults to be a calculated value
        that ensures the word can be cut out.
        '''
        return jieba.add_word(word, freq, tag)

    @classmethod
    def delWord(cls, word):
        '''
        Convenient function for deleting a word.
        '''
        return jieba.del_word(word)

    @classmethod
    def suggestFreq(cls, segment, tune=False):
        '''
        Suggest word frequency to force the characters in a word to be
        joined or splitted.

        Parameter:
            - segment : The segments that the word is expected to be cut into,
                        If the word should be treated as a whole, use a str.
                        如 jieba.suggest_freq(('中', '将'), True)
                        jieba.suggest_freq('中将', True)
            - tune : If True, tune the word frequency.

        Note that HMM may affect the final result. If the result doesn't change,
        set HMM=False.
        '''
        return jieba.suggest_freq(segment, tune)
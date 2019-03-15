#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: pyltp_wrapper.py
@time: 2019/3/15 14:24

pyltp 是 语言技术平台（Language Technology Platform, LTP） 的 Python 封装。
https://pyltp.readthedocs.io/zh_CN/latest/api.html#id2
"""
import pyltp
from pyltp import SentenceSplitter
import os
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import Parser


if __name__ == '__main__':
    mode = 4
    LTP_DATA_DIR = 'D:/data/fact_triple_extraction-master/ltp_data'  # ltp模型目录的路径， 路径中不能有中文
    if mode == 4:
        """
        依存句法分析
        """
        par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
        parser = Parser()  # 初始化实例
        parser.load(par_model_path)  # 加载模型

        words = ['元芳', '你', '怎么', '看']
        postags = ['nh', 'r', 'r', 'v']
        arcs = parser.parse(words, postags)  # 句法分析

        print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))#4:SBV	4:SBV	4:ADV	0:HED
        parser.release()  # 释放模型
    if mode ==3:
        """
        词性标注
        """
        pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
        postagger = Postagger()  # 初始化实例
        postagger.load(pos_model_path)  # 加载模型

        words = ['元芳', '你', '怎么', '看']  # 分词结果
        postags = postagger.postag(words)  # 词性标注

        print('\t'.join(postags))#nh	r	r	v
        postagger.release()  # 释放模型
    if mode == 2:
        """
        分词
        """
        cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
        print(cws_model_path)
        segmentor = Segmentor()  # 初始化实例
        segmentor.load(cws_model_path)  # 加载模型
        words = segmentor.segment('元芳你怎么看')  # 分词
        print('\t'.join(words))#元芳	你	怎么	看
        print(type(words))#<class 'pyltp.VectorOfString'>
        print(list(words))#['元芳', '你', '怎么', '看']
        words = segmentor.segment('二甲双胍是一种化学物质')  # 分词
        print('\t'.join(words))#二	甲	双胍	是	一	种	化学	物质
        #使用外部词典，无效！
        segmentor.load_with_lexicon(cws_model_path, 'd:/words.dic')
        words = segmentor.segment('二甲双胍是一种化学物质')  # 分词
        print('\t'.join(words))  # 二	甲	双胍	是	一	种	化学	物质
        segmentor.release()  # 释放模型
    if mode == 1:
        sents = SentenceSplitter.split('元芳你怎么看？我就趴窗口上看呗！')  # 分句
        print('\n'.join(sents))
        """
        元芳你怎么看？
        我就趴窗口上看呗！
        """

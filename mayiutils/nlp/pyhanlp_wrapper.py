#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: pyhanlp_wrapper.py
@time: 2019/3/20 13:56

HanLP是由一系列模型与算法组成的工具包，目标是普及自然语言处理在生产环境中的应用。
HanLP具备功能完善、性能高效、架构清晰、语料时新、可自定义的特点；
提供词法分析（中文分词、词性标注、命名实体识别）、句法分析、文本分类和情感分析等功能。

HanLP已经被广泛用于Lucene、Solr、ElasticSearch、Hadoop、Android、Resin等平台，
有大量开源作者开发各种插件与拓展，并且被包装或移植到Python、C#、R、JavaScript等语言上去。

项目主页：https://github.com/hankcs/HanLP

pip install pyhanlp

命令行输入：hanlp serve  可以启动可视化界面服务
"""
from pyhanlp import HanLP


if __name__ == '__main__':
    print(HanLP.segment("今天很开心！"))#[今天/t, 很/d, 开心/a, ！/w]
    # print(HanLP.extractSummary(text, 3))#自动摘要提取
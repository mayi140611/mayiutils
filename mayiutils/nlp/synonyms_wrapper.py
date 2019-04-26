#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: synonyms_wrapper.py
@time: 2019-04-26 17:27

http://www.52nlp.cn/synonyms-中文近义词工具包

Synonyms
    Chinese Synonyms for Natural Language Processing and Understanding.
    最好的中文近义词库。
Synonyms的起源
    最近需要做一个基于知识图谱的检索，但是因为知识图谱中存储的都是标准关键词，所以需要对用户的输入进行标准关键词的匹配。
    目前很缺乏质量好的中文近义词库，于是便考虑使用word2vec训练一个高质量的同义词库将"非标准表述" 映射到 "标准表述"，这就是Synonyms的起源。

具体训练方法
    首先需要语料，我们采用了开放的大规模中文语料——维基百科中文语料。

    （1）下载维基百科中文语料。
    （2）繁简转换。
    （3）分词。
    使用gensim自带的word2vec包进行词向量的训练。
    （1）下载gensim。
    （2）输入分词之后的维基语料进行词向量训练。
    （3）测试训练好的词的近义词。


https://github.com/huyingxi/Synonyms
pip install -U synonyms
"""
import synonyms


if __name__ == '__main__':
    mode = 1
    if mode == 1:
        """
        获取近义词列表及对应的分数
        """
        print(synonyms.nearby('人脸'))
        """
        (['人脸', '图片', '通过观察', '几何图形', '图象', '放大镜', '面孔', '貌似', '奇特', '十分相似'], 
        [1.0, 0.5972837, 0.56848586, 0.53183466, 0.5253439, 0.5240093, 0.52310055, 0.5006411, 0.4851142, 0.39761636])
        """
        print(synonyms.nearby('识别'))
        """
        (['识别', '辨识', '辨别', '辨认', '标识', '鉴别', '标记', '识别系统', '比对', '定位'], 
        [1.0, 0.8722487, 0.76409876, 0.72576123, 0.7029182, 0.68861014, 0.6781318, 0.6638289, 0.5903494, 0.56316423])
        """
        print(synonyms.nearby('NOT_EXIST'))
        """
        ([], [])
        """
        print(synonyms.nearby('毛泽东'))
        """
        (['毛泽东', '邓小平', '刘少奇', '毛主席', '林彪', '周恩来', '胡耀邦', '华国锋', '彭德怀', '中共中央'], 
        [1.0, 0.8725412, 0.8721061, 0.85764945, 0.82601166, 0.82051414, 0.8154685, 0.7820129, 0.77770436, 0.77531785])
        """
    if mode == 2:
        """
        获得两个句子的相似度
        返回值：[0-1]，并且越接近于1代表两个句子越相似。
        """
        sen1 = "旗帜引领方向"
        sen2 = "道路决定命运"
        print(synonyms.compare(sen1, sen2))#0.11
        sen1 = "发生历史性变革"
        sen2 = "取得历史性成就"
        print(synonyms.compare(sen1, sen2))#0.362
        sen1 = "明天天气很好"
        sen2 = "明天天气很差"
        print(synonyms.compare(sen1, sen2))#1.0
        sen1 = "明天天气很好"
        sen2 = "明天天气不错"
        print(synonyms.compare(sen1, sen2))#1.0
        sen1 = "明天天气很好"
        sen2 = "今天天气不错"
        print(synonyms.compare(sen1, sen2))#0.671

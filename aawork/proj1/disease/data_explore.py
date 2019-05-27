#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: abnormal_detection_gaussian.py
@time: 2019-04-24 16:39
"""
import pandas as pd
import re
from mayiutils.db.pymysql_wrapper import PyMysqlWrapper
from mayiutils.file_io.pickle_wrapper import PickleWrapper as picklew
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import shutil
import jieba
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity


def standardize(s):
    """
    字符串标准化
        去除所有空格
        去掉末尾最后一个 的
        小写转大写
        中文字符替换： （），【】：“”’‘；
    :param s:
    :return:
    """

    s = re.sub(r'\s+', '', s)
    s = re.sub(r'的$', '', s)# 去掉末尾最后一个 的
    s = re.sub(r',未特指场所$', '', s)
    s = s.upper()
    s = re.sub(r'（', '(', s)
    s = re.sub(r'）', ')', s)
    s = re.sub(r'，', ',', s)
    s = re.sub(r'：', ':', s)
    s = re.sub(r'【', '[', s)
    s = re.sub(r'】', ']', s)
    s = re.sub(r'“|”|’|‘', '"', s)
    s = re.sub(r'；', ';', s)
    return s


def cal_similarity_by_tfidf(name, threshold=0.9):
    """
    tfidf + cosine distance
    :param name:
    :param threshold:
    :return:
    """
    t = tfidf.transform([tokenizer(name)])
    r = cosine_similarity(t, tfidf_features)
    r = pd.Series(r[0]).sort_values(ascending=False)
    return r

def cal_similarity_by_editdistance(name, threshold):
    """
    计算name的相似度
    :param name:
    :return:
    """
    name = standardize(name)
    length = len(name)
    size = int(length * (1 - threshold))
    namelist = list(dis_name_code_dict.keys())
    fnamelist = list(filter(lambda x: length-size <= len(x) <= length+size, namelist))



def match(code1, name1, threshold=0.9):
    """

    :param code1:
    :param name1:
    :return:
    """
    try:
        name = standardize(name1)
        code = standardize(code1)
    except Exception as e:
        print(code1, name1)
        print(e)
        return
    if name in dis_name_code_dict:
        """
        如果匹配上的话，返回(状态码, 原始code, 原始name, 匹配的字典code, 匹配的name, 匹配标记)
        
        匹配标记
            0：表示系统匹配成功
            -1：表示系统未匹配成功，待人工校核
            1：表示系统未匹配成功，已人工校核
        """
        if code in dis_name_code_dict[name]:
            return 10, code, name, code, name, 0
        else:
            return 11, code, name, dis_name_code_dict[name], name, 0
    r = cal_similarity_by_tfidf(name)
    r1 = r[r > threshold]
    rlist = []
    if not r1.empty:
        for i, v in r1.items():
            print(i, v)
            rlist.append((12, code, name, df.iloc[i, 0], df.iloc[i, 1], -1))
    else:
        """
        如果未能匹配，返回 相同的code对应的name，及达到匹配度的前三个name对应的code
        [
            (状态码, 原始code, 原始name, 原始code, 原始code对应的name, -1),
            (状态码, 原始code, 原始name, 匹配的字典code1, 匹配的name1, -1),
            (状态码, 原始code, 原始name, 匹配的字典code2, 匹配的name2, -1),
            (状态码, 原始code, 原始name, 匹配的字典code3, 匹配的name3, -1),
        ]
        """
        if code in dis_code_name_dict:
            rlist.append((2, code, name, code, dis_code_name_dict[code], -1))

        r1 = r[:5]
        for i, v in r1.items():
            print(i, v)
            rlist.append((2, code, name, df.iloc[i, 0], df.iloc[i, 1], -1))
        if len(rlist) == 0:
            rlist.append((2, code, name, -1, -1, -1))
    return 2, rlist


def tokenizer(x):
    """
    分词
    :param x:
    :return:
    """
    x = re.sub(r'[a-zA-Zαβγδ]+', 'alphabet', x)
    x = re.sub(r'[0-9]+', 'num', x)  # 把数字替换为num
    return ' '.join(jieba.lcut(x))


if __name__ == '__main__':
    mode = 2
    df = pd.read_csv('/Users/luoyonggui/Documents/work/dataset/1/icd10_leapstack.csv')
    # 3bitcode
    df3 = pd.read_excel('/Users/luoyonggui/Documents/work/dataset/1/3bitcode.xls', skiprows=[0])
    df4 = pd.read_excel('/Users/luoyonggui/Documents/work/dataset/1/4bitcode.xls', skiprows=[0]).iloc[:, 1:]
    dis_name_code_dict = picklew.loadFromFile('dis_name_code_dict.pkl')
    dis_code_name_dict = picklew.loadFromFile('dis_code_name_dict.pkl')





    df['diag_name_'] = df['diag_name'].apply(tokenizer)
    # df['diag_name_'] = df['diag_name'].apply(lambda x: ' '.join(jieba.lcut(x)))
    x_test = df['diag_name_']
    print('load stopwords')
    with open('../../mayiutils/nlp/stopwords_zh.dic', encoding='utf8') as f:
        stopwords = [s.strip() for s in f.readlines()]
    print('building tfidf array')
    tfidf = TfidfVectorizer(stop_words=stopwords, token_pattern=r"(?u)\b\w+\b")
    tfidf.fit(x_test)
    tfidf_features = tfidf.transform(x_test).toarray()
    print('building tfidf array completed')

    # print(df.shape)#(41058, 2)
    # print(df.head())
    """
  diag_code                           diag_name
0       V50       轻型货车或篷车乘员在轻型货车或篷车与行人或牲畜碰撞中的损伤
1     V50.0                 轻型货车或篷车司机在非交通事故中的损伤
2   V50.001  轻型货车或篷车司机在轻型货车或篷车与行人或牲畜碰撞中损伤,非交通事故
3     V50.1                 轻型货车或篷车乘客在非交通事故中的损伤
4   V50.101  轻型货车或篷车乘客在轻型货车或篷车与行人或牲畜碰撞中损伤,非交通事故
    """
    # 查看 code是否有重复
    # print(df['diag_code'].unique().shape)#(41058,) 说明code没有重复
    # print(df['diag_name'].unique().shape)
    if mode == 3:
        """
        经过人工校核后的处理
            update dis_code_name_dict
            update dis_name_code_dict
        """
        dft = pd.read_csv('dft.csv', encoding='gbk', index_col=0)
        dft = dft[dft.iloc[:, -1]==1]
        # print(dft.head())
        flag = 0
        for line in dft.itertuples():
            if line[3] not in dis_name_code_dict:
                dis_name_code_dict[line[3]] = [line[4]]
                print('新增dis_name_code_dict：{}:{}'.format(line[3], [line[4]]))
                flag = 1
            if line[4] not in dis_code_name_dict and line[5]:
                dis_code_name_dict[line[4]] = line[5]
                print('新增dis_code_name_dict：{}:{}'.format(line[4], line[5]))
                flag = 1
        if flag == 1:
            if not os.path.exists('backups'):
                os.mkdir('backups')
            if os.path.exists('dis_name_code_dict.pkl') and os.path.exists('dis_code_name_dict.pkl'):
                now = datetime.now()
                nowstr = now.strftime('%Y%m%d%H%M%S')
                shutil.move('dis_name_code_dict.pkl', 'backups/dis_name_code_dict{}.pkl'.format(nowstr))
                shutil.move('dis_code_name_dict.pkl', 'backups/dis_code_name_dict{}.pkl'.format(nowstr))
                print('备份字典：backups/dis_name_code_dict{}.pkl'.format(nowstr))
                print('备份字典：backups/dis_code_name_dict{}.pkl'.format(nowstr))
            picklew.dump2File(dis_name_code_dict, 'dis_name_code_dict.pkl')
            picklew.dump2File(dis_code_name_dict, 'dis_code_name_dict.pkl')

    if mode == 2:
        """
        进行匹配
        """
        dft = df3
        # print(dft.head())

        rlist = []
        for line in dft.itertuples():
            r = match(line[1], line[2])
            # print(r)
            if not r:
                continue
            if r[0] == 2:
                rlist.extend(r[1])
            else:
                rlist.append(r)
        arr = np.array(rlist)
        dft = pd.DataFrame(arr, columns=['status', 'code', 'name', 'match_code', 'match_name', 'match_flag'])
        dft.to_csv('dfttt.csv', encoding='gbk')
    if mode == 1:
        """
        把df标准化后存入mysql
        """
        df['diag_code_s'] = df['diag_code'].apply(standardize)
        df['diag_name_s'] = df['diag_name'].apply(standardize)
        pmw = PyMysqlWrapper(db='dics')
        sqltemplate = """
                INSERT INTO disease_dic SET code = "{}", name = "{}"
                """
        dis_name_code_dict = dict()
        dis_code_name_dict = dict()
        for line in df.itertuples():
            sql = sqltemplate.format(line[3], line[4])
            print(sql)
            dis_code_name_dict[line[3]] = line[4]
            if line[4] in dis_name_code_dict:
                dis_name_code_dict[line[4]].append(line[3])
            else:
                dis_name_code_dict[line[4]] = [line[3]]
            # pmw.executeAndCommit(sql)
        picklew.dump2File(dis_name_code_dict, 'dis_name_code_dict.pkl')
        picklew.dump2File(dis_code_name_dict, 'dis_code_name_dict.pkl')
    if mode == 0:
        """
        测试standardize()
        """
        print(standardize('  轻  型cD货车 或，篷车，【aa】bb：cc“dd”ee’ff‘乘客的（在）畜碰      撞中 损伤,非a交b通事  的的  '))
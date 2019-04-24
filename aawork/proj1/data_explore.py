#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: data_explore.py
@time: 2019-04-24 16:39
"""
import pandas as pd
import re
from mayiutils.db.pymysql_wrapper import PyMysqlWrapper
from mayiutils.pickle_wrapper import PickleWrapper as picklew


def standardize(s):
    """
    字符串标准化
        去除所有空格
        小写转大写
        中文字符替换： （），
    :param s:
    :return:
    """
    s = re.sub(r'\s+', '', s)
    s = s.upper()
    s = re.sub(r'（', '(', s)
    s = re.sub(r'）', ')', s)
    s = re.sub(r'，', ',', s)
    return s


def calsimilarity(name, threshold):
    """
    计算name的相似度
    :param name:
    :return:
    """
    name = standardize(name)
    length = len(name)
    size = int(length * (1 - threshold))
    namelist = list(dis_dict.keys())
    fnamelist = list(filter(lambda x: length-size <= len(x) <= length+size, namelist))



def match(code, name, threshold=0.9):
    """

    :param code:
    :param name:
    :return:
    """
    name = standardize(name)
    code = standardize(code)
    if name in dis_dict:
        if code in dis_dict[name]:
            return 10, code, name
        else:
            return 11, dis_dict[name], name



if __name__ == '__main__':
    mode = 1
    df = pd.read_csv('/Users/luoyonggui/Documents/work/dataset/1/icd10_leapstack.csv')
    dis_dict = picklew.loadFromFile('dis_dict.pkl')
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
    if mode == 2:
        """
        3位代码： code中不包含 .
        """
        def t(x):
            return '.' not in x
        print(df['diag_code'].apply(t).value_counts())
        df3 = df[df['diag_code'].apply(t)]
        print(df3.shape)
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
        dis_dict = dict()
        for line in df.itertuples():
            sql = sqltemplate.format(line[3], line[4])
            print(sql)
            if line[4] in dis_dict:
                dis_dict[line[4]].append(line[3])
            else:
                dis_dict[line[4]] = [line[3]]
            pmw.executeAndCommit(sql)
        picklew.dump2File(dis_dict, 'dis_dict.pkl')
    if mode == 0:
        """
        测试standardize()
        """
        print(standardize('  轻  型cD货车 或，篷车，乘客（在）畜碰      撞中 损伤,非a交b通事  '))
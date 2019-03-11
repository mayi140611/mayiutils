#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: kenlm_wrapper.py
@time: 2019/3/11 13:12
kenlm是一个C++编写的语言模型工具，具有速度快、占用内存小的特点，也提供了Python接口。

b：单字词或者多字词的首字
c：多字词的第二字
d：多字词的第三字
e：多字词的其余部分
"""

import kenlm
import sys
from math import log10

#这里的转移概率是人工总结的，总的来说，就是要降低长词的可能性。
trans = {'bb': 1, 'bc': 0.15, 'cb': 1, 'cd': 0.01, 'db': 1, 'de': 0.01, 'eb': 1, 'ee': 0.001}
trans = {i: log10(j) for i, j in trans.items()}


def viterbi(nodes):
    paths = nodes[0]
    for l in range(1, len(nodes)):
        paths_ = paths
        paths = {}
        for i in nodes[l]:
            nows = {}
            for j in paths_:
                if j[-1]+i in trans:
                    nows[j+i]= paths_[j]+nodes[l][i]+trans[j[-1]+i]
            k = list(nows.values()).index(max(nows.values()))
            paths[list(nows.keys())[k]] = list(nows.values())[k]
    return list(paths.keys())[list(paths.values()).index(max(paths.values()))]


def cp(s):
    return (model.score(' '.join(s), bos=False, eos=False) - model.score(' '.join(s[:-1]), bos=False, eos=False)) or -100.0


def mycut(s):
    # 把s转化为4元组
    # cp(s[i]): s[i]是单字词或者多字词的首字的概率；cp(s[i-1:i+1])是多字词的第二字的概率。。。
    nodes = [{'b':cp(s[i]), 'c':cp(s[i-1:i+1]), 'd':cp(s[i-2:i+1]), 'e':cp(s[i-3:i+1])} for i in range(len(s))]
    tags = viterbi(nodes)
    words = [s[0]]
    for i in range(1, len(s)):
        if tags[i] == 'b':
            words.append(s[i])
        else:
            words[-1] += s[i]
    return words


if __name__ == '__main__':
    modelname = sys.argv[1]
    s = sys.argv[2]
    model = kenlm.Model(modelname)
    print(mycut(s))
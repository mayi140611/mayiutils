#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: text_similarity_wrapper.py
@time: 2019/3/17 20:01
"""
import distance


class TextSimilarityWrapper:
    """
    文本相似度/距离计算
    """
    @classmethod
    def calEditDistance(self, s1, s2, mode=1):
        '''
        通过编辑距离计算字符串相似度，可以是任意字符串word，sentence，text
        :mode: 计算相似的算法选择
        '''
        if mode == 1:
            return distance.levenshtein(s1, s2)
        elif mode == 2:
            matrix = [[i+j for j in range(len(s2) + 1)] for i in range(len(s1) + 1)]
            for i in range(1, len(s1)+1):
                for j in range(1, len(s2)+1):
                    if s1[i-1] == s2[j-1]:
                        d = 0
                    else:
                        d = 1
                    matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1,matrix[i-1][j-1]+d)

            return matrix[len(s1)][len(s2)]


if __name__ == '__main__':
    mode = 2
    if mode == 2:
        """
        提取文档特征
        """
        pass
    if mode == 1:
        """
        编辑距离
        """
        print(TextSimilarityWrapper.calEditDistance('我头疼', '我头很疼', 1))#1
        print(TextSimilarityWrapper.calEditDistance('我头疼', '我头很疼', 2))#1
        print(TextSimilarityWrapper.calEditDistance('我头疼', '我头痛欲裂', 1))#3
        print(TextSimilarityWrapper.calEditDistance('我头疼', '我头痛欲裂', 2))#3
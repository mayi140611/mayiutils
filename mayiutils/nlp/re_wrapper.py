#!/usr/bin/python
# encoding: utf-8


import re
import nltk


class ReWrapper(object):
    """
    主要是对python中的re的相关操作的封装
        re.I: 对大小写不敏感
        re.M: 多行匹配， 影响^和$
        re.S: 使 . 匹配换行在内的所有字符
        re.U: 根据Unicode字符集解析字符。这个标志影响\w, \W, \b, \B
        re.L: 做本地识别匹配
    """
    @classmethod
    def re_show(self, regexp, string, left='{', right='}'):
        '''
        把找到的符合regexp的non-overlapping matches标记出来
        如：
        nltk.re_show('[a-zA-Z]+','12fFdsDFDS3rtG4')#12{fFdsDFDS}3{rtG}4
        '''
        return nltk.re_show(regexp, string, left, right)

    @classmethod
    def findall(self, regexp, string):
        '''
        如果regexp中不包含小括号，如
        re.findall('[a-zA-Z]+','12fFdsDFDS3rtG4')#['fFdsDFDS', 'rtG']
        等价于re.findall('([a-zA-Z]+)','12fFdsDFDS3rtG4')#['fFdsDFDS', 'rtG']
        否则：
        re.findall('(\d)\s+(\d)','12 3fFdsDFDS3 4rtG4')#[('2', '3'), ('3', '4')]
        :return: list
        '''
        return re.findall(regexp, string)

    @classmethod
    def match(cls, pattern, string, flags=0):
        """
        和search的区别是从字符串开头开始匹配，如果开头匹配不到，则返回None
        Try to apply the pattern at the start of the string, returning
        a match object, or None if no match was found.
        :return:
        """
        return re.match(pattern, string, flags)

    @classmethod
    def search(cls, pattern, string, flags=0):
        """
        找出第一个匹配，可以返回匹配的位置
        re.search(r'肚子|小腹|上腹|下腹|腹部|肚|腹','我腹疼啊腹好痛')
            <_sre.SRE_Match object; span=(1, 2), match='腹'>
        re.search(r'肚子|小腹|上腹|下腹|腹部|肚|腹','我腹疼啊腹好痛').span()
            (1, 2)
        re.search(r'肚子|小腹|上腹|下腹|腹部|肚|腹','我腹疼啊腹好痛').group()
            '腹'
        Scan through string looking for a match to the pattern, returning
        a match object, or None if no match was found.
        :return:
        """
        return re.search(pattern, string, flags)

    @classmethod
    def sub(cls, pattern, repl, string, count=0, flags=0):
        """
        字符串替换
        re.subn(r' ','_', '1 2 2 3 4')
            ('1_2_2_3_4', 4)
        re.sub(r' ', '_', '1 2 2 3 4')
            '1_2_2_3_4'
        Return the string obtained by replacing the leftmost
        non-overlapping occurrences of the pattern in string by the
        replacement repl.  repl can be either a string or a callable;
        if a string, backslash escapes in it are processed.  If it is
        a callable, it's passed the match object and must return
        a replacement string to be used.
        :param pattern:
        :param repl:
        :param string:
        :param count: 替换的最大次数
        :param flags:
        :return:
        """
        return re.sub(pattern, repl, string)

    @classmethod
    def split(cls,pattern, string, maxsplit=0, flags=0):
        """
        re.split(r' ', '1 2 2 3 4')
            ['1', '2', '2', '3', '4']
        re.split(r' ', '1 2 2 3 4', maxsplit=2)
            ['1', '2', '2 3 4']
        :param pattern:
        :param string:
        :param maxsplit:
        :param flags:
        :return:
        """
        return re.split(pattern, string, maxsplit)
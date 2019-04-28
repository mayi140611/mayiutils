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
    def getPattern(cls, regex, flags=re.DOTALL):
        '''
        生成pattern对象

        :param regex: 正则表达式, 如 r'\d'
        :param flags:
            re.DOTALL 可以让正则表达式中的点（.）匹配包括换行符在内的任意字符
        :return:
        '''
        return re.compile(regex, flags=flags)

    @classmethod
    def findall(self, regexp, string, flags=0):
        '''
        如果regexp中不包含小括号，如
        re.findall('[a-zA-Z]+','12fFdsDFDS3rtG4')#['fFdsDFDS', 'rtG']
        等价于re.findall('([a-zA-Z]+)','12fFdsDFDS3rtG4')#['fFdsDFDS', 'rtG']
        否则：
        re.findall('(\d)\s+(\d)','12 3fFdsDFDS3 4rtG4')#[('2', '3'), ('3', '4')]
        :return: list
        '''
        return re.findall(regexp, string, flags=flags)

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
    def split(cls, pattern, string, maxsplit=0, flags=0):
        """
        re.split(r' ', '1 2 2 3 4')
            ['1', '2', '2', '3', '4']
        re.split(r' ', '1 2 2 3 4', maxsplit=2)
            ['1', '2', '2 3 4']
        re.split(r'[，。！？、]/[bems]', s)
        :param pattern:
        :param string:
        :param maxsplit:
        :param flags:
        :return:
        """
        return re.split(pattern, string, maxsplit, flags=flags)


if __name__ == '__main__':
    mode = 1
    if mode == 3:
        """
        findall
        """

        print(re.findall('([ⅰ]+)','ⅰ ⅱ'))#['f', 's', 'r']
        print(re.findall('([s|rf]+)','12fFdsDFDS3rtG4'))#['f', 's', 'r']
        print(re.findall('([srf]+)','12fFdsDFDS3rtG4'))#['f', 's', 'r']
        print(re.findall('([a-zA-Z]+)','12fFdsDFDS3rtG4'))#['fFdsDFDS', 'rtG']
        print(re.findall('[a-zA-Z]+','12fFdsDFDS3rtG4'))#['fFdsDFDS', 'rtG']
        nltk.re_show('[a-zA-Z]+', '12fFdsDFDS3rtG4')#12{fFdsDFDS}3{rtG}4
        nltk.re_show('[a-zA-Z]*', '12fFdsDFDS3rtG4')#{}1{}2{fFdsDFDS}3{rtG}4{}
        print(re.subn(r' ','_', '1 2 2 3 4'))#('1_2_2_3_4', 4)
        print(re.sub(r' ', '_', '1 2 2 3 4'))#1_2_2_3_4
        print(re.split(r' ', '1 2 2 3 4', maxsplit=2))#['1', '2', '2 3 4']
    if mode == 2:
        """
        同义词替换 match search
        """
        fubu = ['肚子', '小腹', '上腹', '下腹', '腹部', '肚', '腹']
        jinbu = ['脖子', '头颈', '脖颈', '脖梗', '颈部', '颈椎', '颈', '脖']
        yanhou = ['扁桃体', '嗓子', '喉咙', '咽喉', '食管', '喉结', '声带', '咽', '喉']
        nanxsz = ['男性生殖', '睾丸', '包皮', '阴茎', '阴囊', '鸡鸡', '龟头', '精']
        nvxsz = ['女性生殖', '产褥期', '排卵期', '阴道', '白带', '经期', '月经', '闭经', '会阴', '子宫']
        penqiang = ['尿道', '盆腔', '耻骨', '髋']
        beibu = ['脊柱', '背脊', '背部', '背']
        tunbu = ['肛门', '臀部', '屁股', '屁', '臀', '髋']
        yaobu = ['肾区', '腰部', '腰']
        daxiaobian = ['大小便', '大便', '小便', '屎', '尿']
        pattern = re.compile(r'|'.join(fubu))
        nltk.re_show(r'肚子|小腹|上腹|下腹|腹部|肚|腹', '我肚子疼')#我{肚子}疼
        nltk.re_show(r'肚子|小腹|上腹|下腹|腹部|肚|腹', '我腹部疼')#我{腹部}疼
        nltk.re_show(r'肚子|小腹|上腹|下腹|腹部|肚|腹', '我腹疼')#我{腹}疼
        r = re.search(r'肚子|小腹|上腹|下腹|腹部|肚|腹', '我腹疼啊腹好痛')
        print(r)#<_sre.SRE_Match object; span=(1, 2), match='腹'>
        print(r.span())#(1, 2)
        print(r.group())#腹
        r = re.match(r'肚子|小腹|上腹|下腹|腹部|肚|腹', '我腹疼啊腹好痛')
        print(r)#None
        print(re.match(r'肚子|小腹|上腹|下腹|腹部|肚|腹','腹疼啊腹好痛'))#<_sre.SRE_Match object; span=(0, 1), match='腹'>
    if mode == 1:
        """
        https://www.runoob.com/python/python-reg-expressions.html
        \w	匹配字母数字及下划线
            注意：中文的标的符号也不会被匹配
        """
        nltk.re_show(r'[a-zA-Z]{1,2}', '12fFdsDFDS3rtG4')#12{fF}{ds}{DF}{DS}3{rt}{G}4
        nltk.re_show(r'\w+', '12fFdsD.FDS3rtG4')#{12fFdsD}.{FDS3rtG4}
        nltk.re_show(r'\w+', 'ian说：你好，我（ian）是ian_luo。hi, nice to meet u!')#{ian说}：{你好}，{我}（{ian}）{是ian_luo}。{hi}, {nice} {to} {meet} {u}!

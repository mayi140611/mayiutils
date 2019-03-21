#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: auto_abstract.py
@time: 2019/3/20 14:01

自动摘要: 通过手动处理文章的方式，效果肯定不会好
"""
import jieba,copy,re,codecs
from collections import Counter

from apps.auto_title.text import title, text
from pyhanlp import HanLP


class Summary():
    #**** 切分句子 ************
    def cutSentence(self,text):
        sents = []
        text = re.sub(r'\n+','。',text)  # 换行改成句号（标题段无句号的情况）
        text = text.replace('。。','。')  # 删除多余的句号
        text = text.replace('？。','。')  #
        text = text.replace('！。','。')  # 删除多余的句号
        sentences = re.split(r'。|！|？|】|；',text) # 分句
        #print(sentences)
        sentences = sentences[:-1] # 删除最后一个句号后面的空句
        for sent in sentences:
            len_sent = len(sent)
            if len_sent < 4:  # 删除换行符、一个字符等
                continue
            # sent = sent.decode('utf8')
            sent = sent.strip('　 ')
            sent = sent.lstrip('【')
            sents.append(sent)
        return sents

    #**** 提取特征词 **********************
    def getKeywords(self,title,sentences,n=10):
        words = []
        #**** 分词，获取词汇列表 *****
        # split_result = pseg.cut(text)
        for sentence in sentences:
            split_result = jieba.cut(sentence)
            for i in split_result:
                words.append(i)
        #**** 统计词频TF *****
        c = Counter(words) # 词典
        #**** 去除停用词(为了提高效率，该步骤放到统计词频之后)
        self.delStopwords(c)
        #**** 标题中提取特征 *********
        words_title = [word for word in jieba.cut(title,cut_all=True)]
        self.delStopwords(words_title)
        #**** 获取topN ************
        topN = c.most_common(n)
        # for i in topN:
        #     print(i[0],i[1])
        words_topN = [i[0] for i in topN if i[1]>1] #在topN中排除出现次数少于2次的词

        words_topN = list(set(words_topN)|set(words_title)) # 正文关键词与标题关键词取并集

        print (' '.join(words_topN))
        return words_topN

    #**** 去除停用词 *******************************
    def delStopwords(self,dict):
        sw_file = codecs.open('../../mayiutils/nlp/stopwords_zh.dic', encoding='utf8')
        stop_words = []
        for line in sw_file.readlines():
            stop_words.append(line.strip())
        #***** 输入参数为list *************
        # if type(dict) is types.ListType:
        if type(dict) is list:
            words = dict
            for word in words:
                if word in stop_words:
                    words.remove(word)
        #***** 输入参数type为 <class 'collections.Counter'>  *****
        else:
            words = copy.deepcopy(list(dict.keys()))
            for word in words:
                if word in stop_words:
                    del dict[word]
        return words

    #**** 提取topN句子 **********************
    def getTopNSentences(self,sentences,keywords,n=3):
        sents_score = {}
        len_sentences = len(sentences)
        #**** 初始化句子重要性得分，并计算句子平均长度
        len_avg = 0
        len_min = len(sentences[0])
        len_max = len(sentences[0])
        for sent in sentences:
            sents_score[sent] = 0
            l = len(sent)
            len_avg += l
            if len_min > l:
                len_min = l
            if len_max < l:
                len_max = l
        len_avg = len_avg / len_sentences
        # print(len_min,len_avg,len_max)
        #**** 计算句子权重得分 **********
        for sent in sentences:
            #**** 不考虑句长在指定范围外的句子 ******
            l = len(sent)
            if l < (len_min + len_avg) / 2 or l > (3 * len_max - 2 * len_avg) / 4:
                continue
            words = []
            sent_words = jieba.cut(sent) # <generator object cut at 0x11B38120>
            for i in sent_words:
                words.append(i)
            keywords_cnt = 0
            len_sent = len(words)
            if len_sent == 0:
                continue

            for word in words:
                if word in keywords:
                    keywords_cnt += 1
            score = keywords_cnt * keywords_cnt * 1.0 / len_sent
            sents_score[sent] = score
            if sentences.index(sent) == 0:# 提高首句权重
                sents_score[sent] = 2 * score
        #**** 排序 **********************
        dict_list = sorted(sents_score.items(),key=lambda x:x[1],reverse=True)
        # print(dict_list)
        #**** 返回topN ******************
        sents_topN = []
        for i in dict_list[:n]:
            sents_topN.append(i[0])
            # print i[0],i[1]
        sents_topN = list(set(sents_topN))
        #**** 按比例提取 **************************
        if len_sentences <= 5:
            sents_topN = sents_topN[:1]
        elif len_sentences < 9:
            sents_topN = sents_topN[:2]

        return sents_topN

    #**** 恢复topN句子在文中的相对顺序 *********
    def sents_sort(self,sents_topN,sentences):
        keysents = []
        for sent in sentences:
            if sent in sents_topN and sent not in keysents:
                keysents.append(sent)
        keysents = self.post_processing(keysents)

        return keysents

    def post_processing(self,keysents):
        #**** 删除不完整句子中的详细部分 ********************
        detail_tags = ['，一是','：一是','，第一，','：第一，','，首先，','；首先，']
        for i in keysents:
            for tag in detail_tags:
                index = i.find(tag)
                if index != -1:
                    keysents[keysents.index(i)] = i[:index]
        #**** 删除编号 ****************************
        for i in keysents:
            # print(i)
            regex = re.compile(r'^一、|^二、|^三、|^三、|^四、|^五、|^六、|^七、|^八、|^九、|^十、|^\d{1,2}、|^\d{1,2} ')
            result = re.findall(regex,i)
            if result:
                keysents[keysents.index(i)] = re.sub(regex,'',i)
        #**** 删除备注性质的句子 ********************
        for i in keysents:
            regex = re.compile(r'^注\d*：')
            result = re.findall(regex,i)
            if result:
                keysents.remove(i)
        #**** 删除句首括号中的内容 ********************
        for i in keysents:
            regex = re.compile(r'^\[.*\]')
            result = re.findall(regex,i)
            if result:
                keysents[keysents.index(i)] = re.sub(regex,'',i)
        #**** 删除来源(空格前的部分) ********************
        for i in keysents:
            regex = re.compile(r'^.{1,20} ')
            result = re.findall(regex,i)
            if result:
                keysents[keysents.index(i)] = re.sub(regex,'',i)
        #**** 删除引号部分（如：银行间债市小幅下跌，见下图：） ********************
        for i in keysents:
            regex = re.compile(r'，[^，]+：$')
            result = re.findall(regex,i)
            if result:
                keysents[keysents.index(i)] = re.sub(regex,'',i)

        return keysents

    def main(self,title,text):
        sentences = self.cutSentence(text)
        keywords = self.getKeywords(title, sentences, n=8)
        sents_topN = self.getTopNSentences(sentences, keywords, n=3)
        keysents = self.sents_sort(sents_topN, sentences)
        print(keysents)
        return keysents


if __name__=='__main__':
    summary=Summary()
    summary.main(title, text)
    """
    ['这篇文章将描绘一下Kensho、文因互联、数库科技、通联数据在这个领域的探索和尝试,看看新时代正在掀起的巨浪', '他迅速构建起庞大的金融帝国:浙商银行、浙商基金、民生人寿、万向财务、通联数据等公司',
     '2017年6月6日,恒生电子正式面向金融机构推出最新的人工智能产品:涵盖智能投资、智能资讯、智能投顾、智能客服四大领域']
    """
    print(HanLP.extractSummary(text, 3))
    """
    [大数据、算法驱动的人工智能已经进入到金融领域, 这篇文章将描绘一下Kensho、文因互联、数库科技、通联数据在这个领域的探索和尝试, 
    华尔街的Kensho是金融数据分析领域里谁也绕不过的一个独角兽]
    """
    s2 = '8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'
    print(HanLP.extractSummary(s2, 3))
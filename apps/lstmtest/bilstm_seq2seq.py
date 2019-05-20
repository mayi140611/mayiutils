#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: bilstm_seq2seq.py
@time: 2019/3/11 17:02
"""
import re
import numpy as np
import pandas as pd
from mayiutils.file_io.pickle_wrapper import PickleWrapper as picklew
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from tensorflow.keras.models import Model, load_model


def clean(s): #整理一下数据，有些不规范的地方
    if '“/s' not in s:
        return s.replace(' ”/s', '')
    elif '”/s' not in s:
        return s.replace('“/s ', '')
    elif '‘/s' not in s:
        return s.replace(' ’/s', '')
    elif '’/s' not in s:
        return s.replace('‘/s ', '')
    else:
        return s


def get_xy(s):
    """
    获取word序列和label序列

    :param s:
    :return:
        (['“', '人', '们', '常', '说', '生', '活', '是', '一', '部', '教', '科', '书'],
        ['s', 'b', 'e', 's', 's', 'b', 'e', 's', 's', 's', 'b', 'm', 'e'])
    """
    s = re.findall('(.)/(.)', s)
    # print(s)
    if s:
        s = np.array(s)
        return list(s[:, 0]), list(s[:, 1])


def trans_one(x):
    """
    把label ['s', 'b'...]转换为one-hot形式
    :param x:
    :return:
    """
    _ = map(lambda y: tf.keras.utils.to_categorical(y,5), tag[x].values.reshape((-1,1)))
    _ = list(_)
    _.extend([np.array([[0,0,0,0,1]])]*(maxlen-len(x)))
    return np.array(_)
#转移概率，单纯用了等概率
zy = {'be':0.5,
      'bm':0.5,
      'eb':0.5,
      'es':0.5,
      'me':0.5,
      'mm':0.5,
      'sb':0.5,
      'ss':0.5
     }

zy = {i:np.log(zy[i]) for i in zy.keys()}


def viterbi(nodes):
    paths = {'b': nodes[0]['b'], 's': nodes[0]['s']}
    for l in range(1,len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1]+i in zy.keys():
                    nows[j+i] = paths_[j]+nodes[l][i]+zy[j[-1]+i]
            k = np.argmax(list(nows.values()))
            paths[list(nows.keys())[k]] = list(nows.values())[k]
    return list(paths.keys())[np.argmax(paths.values())]


def simple_cut(s):
    if s:
        r = model.predict(np.array([list(chars[list(s)].fillna(0).astype(int))+[0]*(maxlen-len(s))]), verbose=False)[0][:len(s)]
        # print(type(r), r.shape, r[:2])
        # return
        r = np.log(r)
        nodes = [dict(zip(['s', 'b', 'm', 'e'], i[:4])) for i in r]
        t = viterbi(nodes)
        words = []
        for i in range(len(s)):
            if t[i] in ['s', 'b']:
                words.append(s[i])
            else:
                words[-1] += s[i]
        return words
    else:
        return []


not_cuts = re.compile(r'([\da-zA-Z ]+)|[。，、？！.?,!]')


def cut_word(s):
    result = []
    j = 0
    for i in not_cuts.finditer(s):
        result.extend(simple_cut(s[j:i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(simple_cut(s[j:]))
    return result


if __name__ == '__main__':
    mode = 2
    chars = picklew.loadFromFile('chars.pkl')
    maxlen = 32
    if mode == 2:
        model = load_model('model.h5')
        simple_cut('苏剑林是科学空间的博主')
        print(cut_word('苏剑林是科学空间的博主'))
        print(cut_word('你是真的遇到过报错了'))
        print(cut_word('列夫·托尔斯泰是俄罗斯一位著名的作家'))
    if mode == 1:
        """
        train model
        """
        s = open('msr_train.txt', encoding='gbk').read()
        s = s.split('\r\n')
        # print(s[0])
        s = ''.join(map(clean, s))
        s = re.split(r'[，。！？、]/[bems]', s)
        print(s[0])
        data = []  # 生成训练样本
        label = []
        for i in s:
            x = get_xy(i)
            if x:
                data.append(x[0])
                label.append(x[1])

        d = pd.DataFrame(index=range(len(data)))
        d['data'] = data
        d['label'] = label
        # print(d.head())
        """
        抛弃了多于32字的样本，这部分样本很少，事实上，用逗号、句号等天然分隔符分开后，句子很少有多于32字的。
        """

        d = d[d['data'].apply(len) <= maxlen]
        d.index = range(len(d))
        """
        这次我用了5tag，在原来的4tag的基础上，加上了一个x标签，
        用来表示不够32字的部分，比如句子是20字的，那么第21～32个标签均为x。
        """
        tag = pd.Series({'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4})
        chars = []  # 统计所有字，跟每个字编号
        for i in data:
            chars.extend(i)
        # 按照词频出现的高低给word编号
        chars = pd.Series(chars).value_counts().sort_values(ascending=False)
        chars[:] = range(1, len(chars) + 1)
        picklew.dump2File(chars, 'chars.pkl')
        # # 生成适合模型输入的格式
        # d['x'] = d['data'].apply(lambda x: np.array(list(chars[x]) + [0] * (maxlen - len(x))))
        #
        # d['y'] = d['label'].apply(trans_one)

        # picklew.dump2File(d, 'd.pkl')
        d = picklew.loadFromFile('d.pkl')
        # 设计模型
        word_size = 128
        maxlen = 32

        sequence = Input(shape=(maxlen,), dtype='int32')
        embedded = Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=True)(sequence)
        blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
        output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
        model = Model(inputs=sequence, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(model.summary())
        """
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        input_1 (InputLayer)         (None, 32)                0         
        _________________________________________________________________
        embedding (Embedding)        (None, 32, 128)           660864    
        _________________________________________________________________
        bidirectional (Bidirectional (None, 32, 64)            98816     
        _________________________________________________________________
        time_distributed (TimeDistri (None, 32, 5)             325       
        =================================================================
        Total params: 760,005
        Trainable params: 760,005
        Non-trainable params: 0
        _________________________________________________________________
        None
        """
        batch_size = 1024
        history = model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1, maxlen, 5)), batch_size=batch_size,
                            nb_epoch=50)
        model.save('model.h5')











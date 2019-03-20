#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: seq2seq1.py
@time: 2019/3/19 19:42

https://spaces.ac.cn/archives/5861
"""
import numpy as np
from tqdm import tqdm
import os,json
# import uniout
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
import xml.sax
import re
# from mayiutils.pickle_wrapper import PickleWrapper as picklew
import pickle


def loadFromFile(file, mode='rb'):
    with open(file, mode) as f:
        return pickle.load(f)


def dump2File(o, file, mode='wb'):
    '''
    把目标对象序列化到文件
    :param o: 目标对象
    :param file: 文件
    :param mode:
    :return:
    '''
    with open(file, mode) as f:
        pickle.dump(o, f)


def str2id(s, start_end=False):
    # 文字转整数id
    if start_end: # 补上<start>和<end>标记
        ids = [char2id.get(c, 1) for c in s[:maxlen-2]]
        ids = [2] + ids + [3]
    else: # 普通转化
        ids = [char2id.get(c, 1) for c in s[:maxlen]]
    return ids


def id2str(ids):
    # id转文字，找不到的用空字符代替
    return ''.join([id2char.get(i, '') for i in ids])


def padding(x):
    # padding至batch内的最大长度
    ml = max([len(i) for i in x])
    return [i + [0] * (ml-len(i)) for i in x]


def data_generator():
    # 数据生成器
    X, Y = [], []
    while True:
        for a in rlist:
            X.append(str2id(a[1]))
            Y.append(str2id(a[0], start_end=True))
            if len(X) == batch_size:
                X = np.array(padding(X))
                Y = np.array(padding(Y))
                yield [X, Y], None
                X, Y = [], []


def to_one_hot(x):
    """
    输出一个词表大小的向量，来标记该词是否在文章出现过
    """
    x, x_mask = x
    x = K.cast(x, 'int32')
    x = K.one_hot(x, len(chars)+4)
    x = K.sum(x_mask * x, 1, keepdims=True)
    x = K.cast(K.greater(x, 0.5), 'float32')
    return x


class ScaleShift(Layer):
    """
    缩放平移变换层（Scale and shift）
    """
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = (1,)*(len(input_shape)-1) + (input_shape[-1],)
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')

    def call(self, inputs):
        x_outs = K.exp(self.log_scale) * inputs + self.shift
        return x_outs


class Interact(Layer):
    """
    交互层，负责融合encoder和decoder的信息
    """
    def __init__(self, **kwargs):
        super(Interact, self).__init__(**kwargs)

    def build(self, input_shape):
        in_dim = input_shape[0][-1]
        out_dim = input_shape[1][-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(in_dim, out_dim),
                                      initializer='glorot_normal')

    def call(self, inputs):
        q, v, v_mask = inputs
        k = v
        mv = K.max(v - (1. - v_mask) * 1e10, axis=1, keepdims=True) # maxpooling1d
        mv = mv + K.zeros_like(q[:,:,:1]) # 将mv重复至“q的timesteps”份
        # 下面几步只是实现了一个乘性attention
        qw = K.dot(q, self.kernel)
        a = K.batch_dot(qw, k, [2, 2]) / 10.
        a -= (1. - K.permute_dimensions(v_mask, [0, 2, 1])) * 1e10
        a = K.softmax(a)
        o = K.batch_dot(a, v, [2, 1])
        # 将各步结果拼接
        return K.concatenate([o, q, mv], 2)

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1],
                input_shape[0][2]+input_shape[1][2]*2)


def gen_title(s, topk=3):
    """
    beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    xid = np.array([str2id(s)] * topk) # 输入转id
    yid = np.array([[2]] * topk) # 解码均以<start>开通，这里<start>的id为2
    scores = [0] * topk # 候选答案分数
    for i in range(50): # 强制要求标题不超过50字
        proba = model.predict([xid, yid])[:, i, 3:] # 直接忽略<padding>、<unk>、<start>
        log_proba = np.log(proba + 1e-6) # 取对数，方便计算
        arg_topk = log_proba.argsort(axis=1)[:,-topk:] # 每一项选出topk
        _yid = [] # 暂存的候选目标序列
        _scores = [] # 暂存的候选目标序列得分
        if i == 0:
            for j in range(topk):
                _yid.append(list(yid[j]) + [arg_topk[0][j]+3])
                _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
        else:
            for j in range(len(xid)):
                for k in range(topk): # 遍历topk*topk的组合
                    _yid.append(list(yid[j]) + [arg_topk[j][k]+3])
                    _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            _arg_topk = np.argsort(_scores)[-topk:] # 从中选出新的topk
            _yid = [_yid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        yid = []
        scores = []
        for k in range(len(xid)):
            if _yid[k][-1] == 3: # 找到<end>就返回
                return id2str(_yid[k])
            else:
                yid.append(_yid[k])
                scores.append(_scores[k])
        yid = np.array(yid)
    # 如果50字都找不到<end>，直接返回
    return id2str(yid[np.argmax(scores)])


class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 训练过程中观察一两个例子，显示标题质量提高的过程
        print(gen_title(s1))
        print(gen_title(s2))
        # 保存最优结果
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')


if __name__ == '__main__':
    mode = 2
    min_count = 32
    maxlen = 400
    batch_size = 64
    epochs = 100
    char_size = 128

    # 提取文章标题和正文内容
    if mode == 1:
        file_name = os.listdir('D:/Desktop/SogouCS.reduced')
        rlist = []
        for fn in file_name:
            f = open(os.path.join('D:/Desktop/SogouCS.reduced', fn), 'r', encoding='utf8')
            r = f.read()
            f.close()
            r = re.sub(r'\u3000', ' ', r)#去除中文空格
            pattern = re.compile(r'<contenttitle>(.*?)</contenttitle>.*?<content>(.*?)</content>', flags=re.DOTALL)
            rr = re.findall(pattern, r)
            print(len(rr))
            rr = [i for i in rr if i[0].strip()!='' and i[1].strip()!='']
            print(len(rr))
            rlist.extend(rr)
            # print(rr[0])
        print(len(rr))
        print(rr[0])
        dump2File(rlist, 'rlist.pkl')

    rlist = loadFromFile('rlist.pkl')

    if os.path.exists('seq2seq_config.json'):
        chars, id2char, char2id = json.load(open('seq2seq_config.json'))
        id2char = {int(i): j for i, j in id2char.items()}
    else:
        # 统计字频，给字典编号
        chars = {}
        for a in tqdm(rlist):
            for w in a[1]:  # 纯文本，不用分词
                chars[w] = chars.get(w, 0) + 1
            for w in a[0]:  # 纯文本，不用分词
                chars[w] = chars.get(w, 0) + 1
        chars = {i: j for i, j in chars.items() if j >= min_count}
        # 0: mask
        # 1: unk
        # 2: start
        # 3: end
        id2char = {i + 4: j for i, j in enumerate(chars)}
        char2id = {j: i for i, j in id2char.items()}
        json.dump([chars, id2char, char2id], open('seq2seq_config.json', 'w'))

    # 搭建seq2seq模型

    x_in = Input(shape=(None,))
    y_in = Input(shape=(None,))
    x = x_in
    y = y_in
    # expand_dims: Adds a 1-sized dimension at index "axis"
    x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
    y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(y)

    x_one_hot = Lambda(to_one_hot)([x, x_mask])
    x_prior = ScaleShift()(x_one_hot)  # 学习输出的先验分布（标题的字词很可能在文章出现过）

    embedding = Embedding(len(chars) + 4, char_size)
    x = embedding(x)
    y = embedding(y)

    # encoder，双层双向LSTM
    x = Bidirectional(LSTM(char_size // 2, return_sequences=True))(x)
    x = Bidirectional(LSTM(char_size // 2, return_sequences=True))(x)

    # decoder，双层单向LSTM
    y = LSTM(char_size, return_sequences=True)(y)
    y = LSTM(char_size, return_sequences=True)(y)

    xy = Interact()([y, x, x_mask])
    xy = Dense(512, activation='relu')(xy)
    xy = Dense(len(chars) + 4)(xy)
    xy = Lambda(lambda x: (x[0] + x[1]) / 2)([xy, x_prior])  # 与先验结果平均
    xy = Activation('softmax')(xy)

    # 交叉熵作为loss，但mask掉padding部分
    cross_entropy = K.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
    loss = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])

    model = Model([x_in, y_in], xy)
    model.add_loss(loss)
    model.compile(optimizer=Adam(1e-3))
    print(model.summary())
    """
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_2 (InputLayer)            (None, None)         0                                            
    __________________________________________________________________________________________________
    input_1 (InputLayer)            (None, None)         0                                            
    __________________________________________________________________________________________________
    embedding_1 (Embedding)         (None, None, 128)    843392      input_1[0][0]                    
                                                                     input_2[0][0]                    
    __________________________________________________________________________________________________
    lstm_3 (LSTM)                   (None, None, 128)    131584      embedding_1[1][0]                
    __________________________________________________________________________________________________
    bidirectional_1 (Bidirectional) (None, None, 128)    98816       embedding_1[0][0]                
    __________________________________________________________________________________________________
    lstm_4 (LSTM)                   (None, None, 128)    131584      lstm_3[0][0]                     
    __________________________________________________________________________________________________
    bidirectional_2 (Bidirectional) (None, None, 128)    98816       bidirectional_1[0][0]            
    __________________________________________________________________________________________________
    lambda_1 (Lambda)               (None, None, 1)      0           input_1[0][0]                    
    __________________________________________________________________________________________________
    interact_1 (Interact)           (None, None, 384)    16384       lstm_4[0][0]                     
                                                                     bidirectional_2[0][0]            
                                                                     lambda_1[0][0]                   
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, None, 512)    197120      interact_1[0][0]                 
    __________________________________________________________________________________________________
    lambda_3 (Lambda)               (None, 1, 6589)      0           input_1[0][0]                    
                                                                     lambda_1[0][0]                   
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, None, 6589)   3380157     dense_1[0][0]                    
    __________________________________________________________________________________________________
    scale_shift_1 (ScaleShift)      (None, 1, 6589)      13178       lambda_3[0][0]                   
    __________________________________________________________________________________________________
    lambda_4 (Lambda)               (None, None, 6589)   0           dense_2[0][0]                    
                                                                     scale_shift_1[0][0]              
    __________________________________________________________________________________________________
    activation_1 (Activation)       (None, None, 6589)   0           lambda_4[0][0]                   
    ==================================================================================================
    Total params: 4,911,031
    Trainable params: 4,911,031
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None
    """
    s1 = '夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医。'
    s2 = '8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'

    evaluator = Evaluate()

    model.fit_generator(data_generator(),
                        steps_per_epoch=1000,
                        epochs=epochs,
                        callbacks=[evaluator])
    model.save('auto_title_model0320.h5')
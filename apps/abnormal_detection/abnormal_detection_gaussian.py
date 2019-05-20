#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: abnormal_detection_gaussian.py
@time: 2019/4/14 10:29
"""
import pandas as pd
import numpy as np
import math
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.model_selection import KFold, train_test_split


def prob(x):
    """
    x是行向量
    :param x:
    :return:
    """
    p = 1 / ((2 * math.pi) ** (train_set.shape[1] * 0.5) * np.linalg.det(sigma) ** 0.5) \
        * math.exp(-0.5 * (x - mu).dot(np.linalg.inv(sigma)).dot((x - mu).T))
    return p


if __name__ == "__main__":
    mode = 3
    df = pd.read_csv('../../tmp/creditcard.csv')
    """
    对于严重倾斜的样本，如何划分训练集、验证集、测试集？
    https://www.bilibili.com/video/av9912938/?p=92
    1、先把正常positive样本和异常negtive样本区分开来
    2、然后把正常样本分成5份
        拿出3份作为train_set
        拿出1份+异常样本作为cv_set
        拿出1份+异常样本作为test_set
        
    如何评定最优？https://www.bilibili.com/video/av9912938/?p=92
        f1值最大
        AUC最大？？？ 不合适，应该是理解没有到位
        AUPRC (area under the precision-recall curve) as an evaluation metric
            AUPRC has advantages over AUC when the class distribution is highly skewed（倾斜. 
            The trapezoidal method can be used to approximate the area under the curve.
    """
    # 获取正常样本和异常样本
    positive_set = df[df['Class'] == 0].values[:, 1:-1]
    negtive_set = df[df['Class'] == 1].values[:, 1:-1]
    data_set, test_set = train_test_split(positive_set, test_size=0.2, random_state=14)
    kf = KFold(n_splits=4, shuffle=True, random_state=14)
    if mode == 3:
        """
        利用多元高斯分布求解异常监测2 假设特征间相关
        这是更常见的一种情况！！！
        https://www.bilibili.com/video/av9912938/?p=95
        
        步骤：
        1、参数估计
            mu：n维向量，n为特征数
            sigma：协方差矩阵。n * n维矩阵
        2、求出联合概率分布
        3、确定最优阈值
        """
        #3，利用网格法，确定最优阈值epsilon
        epsilons = [3.1598899884113835e-23, 3.1598899884113835e-22, 3.1598899884113835e-21]
        """
        [0.19844890510948907, 0.1996328591096833, 0.19402319357716324, 0.1963882618510158] 0.19712330491183785
        [0.172800629797284, 0.17192089289210888, 0.1681992337164751, 0.17018802093428959] 0.1707771943350394
        [0.14621237181607677, 0.14525139664804468, 0.1431340872374798, 0.1439036301481361] 0.14462537146243434
        """
        epsilons = [1.1598899884113835e-25, 1.1598899884113835e-27, 1.1598899884113835e-29]
        """
        [0.2525073746312685, 0.2519823788546256, 0.2476905311778291, 0.2502187226596675] 0.2505997518308477
        [0.29037742264535876, 0.29018008834522596, 0.28409846972721225, 0.28619302949061665] 0.2877122525521034
                     precision    recall  f1-score   support
        
                  0       1.00      0.97      0.98     56863
                  1       0.19      0.84      0.31       492
        
        avg / total       0.99      0.97      0.98     57355
        [0.31051253273475493, 0.31439393939393934, 0.3074648928307465, 0.3083209509658246] 0.31017307898131635        
        """
        epsilons = [1.1598899884113835e-31, 1.1598899884113835e-34, 1.1598899884113835e-37]
        """
        [0.3387755102040816, 0.3412828947368421, 0.3323988786543852, 0.3384991843393148] 0.33773911698365594
        [0.36868008948545866, 0.37780834479596515, 0.3624396665204037, 0.3691756272401434] 0.3695259320104927
                     precision    recall  f1-score   support
        
                  0       1.00      0.98      0.99     56863
                  1       0.26      0.84      0.40       492
        
        avg / total       0.99      0.98      0.98     57355
        
        [0.4005847953216374, 0.40997506234413966, 0.39217557251908397, 0.3957631198844487] 0.39962463751732746        
        """
        epsilons = [1.1598899884113835e-40, 1.1598899884113835e-44, 1.1598899884113835e-48]
        """
        [0.4328593996840442, 0.4356120826709062, 0.418960244648318, 0.42524573202276256] 0.42816936475650774
        [0.46249294980259453, 0.47289504036908886, 0.4566666666666667, 0.4610207515423443] 0.4632688520951736
                     precision    recall  f1-score   support
        
                  0       1.00      0.99      0.99     56863
                  1       0.34      0.83      0.48       492
        
        avg / total       0.99      0.98      0.99     57355
        [0.4845238095238096, 0.49333333333333323, 0.48366013071895425, 0.480565371024735] 0.48552066115020803        
        """
        epsilons = [1.1598899884113835e-52, 1.1598899884113835e-56, 1.1598899884113835e-60]
        """
        [0.512950094756791, 0.5155555555555555, 0.5062344139650873, 0.5043478260869565] 0.5097719725910976
        [0.5388188453881885, 0.5453324378777703, 0.529335071707953, 0.5296803652968037] 0.5357916800676789
                     precision    recall  f1-score   support
        
                  0       1.00      0.99      0.99     56863
                  1       0.41      0.81      0.55       492
        
        avg / total       0.99      0.99      0.99     57355
        [0.5606166783461808, 0.5645730416372619, 0.5468215994531784, 0.5460750853242321] 0.5545216011902133
        """
        epsilons = [1.1598899884113835e-62, 1.1598899884113835e-66, 1.1598899884113835e-70]
        """
        [0.5706134094151212, 0.5749279538904899, 0.5563282336578582, 0.5544005544005544] 0.5640675378410058
        [0.5855457227138643, 0.5908750934928946, 0.577259475218659, 0.5679542203147354] 0.5804086279350382
                     precision    recall  f1-score   support
        
                  0       1.00      0.99      1.00     56863
                  1       0.46      0.80      0.59       492
        
        avg / total       0.99      0.99      0.99     57355
        [0.6039755351681957, 0.6113266097750194, 0.6016755521706018, 0.5892857142857143] 0.6015658528498828        
        """
        epsilons = [1.1598899884113835e-80, 1.1598899884113835e-85, 1.1598899884113835e-90]
        """
        [0.6431501230516817, 0.6538461538461539, 0.629718875502008, 0.627294493216281] 0.6385024114040312
        [0.6621621621621622, 0.6689595872742906, 0.640329218106996, 0.6457990115321252] 0.6543124947688935
                     precision    recall  f1-score   support
                  0       1.00      0.99      1.00     56863
                  1       0.57      0.79      0.66       492
        avg / total       0.99      0.99      0.99     57355
        [0.6712683347713546, 0.6843501326259946, 0.6609880749574106, 0.6621507197290433] 0.6696893155209507
        """
        epsilons = [1.1598899884113835e-100, 1.1598899884113835e-110, 1.1598899884113835e-120]
        """
        [0.6912691269126913, 0.6977168949771689, 0.6826241134751773, 0.6825817860300619] 0.6885479803487748
        [0.7180451127819547, 0.7149576669802445, 0.6978021978021978, 0.7079482439926064] 0.7096883053892509
                     precision    recall  f1-score   support
                  0       1.00      1.00      1.00     56863
                  1       0.67      0.77      0.72       492
        avg / total       1.00      0.99      0.99     57355
        [0.7312859884836852, 0.709551656920078, 0.7091254752851711, 0.7175141242937851] 0.7168693112456799
        """
        epsilons = [1.1598899884113835e-125, 1.1598899884113835e-130, 1.1598899884113835e-135]
        """
        [0.7256809338521402, 0.7091633466135459, 0.7080504364694471, 0.7239732569245464] 0.7167169934649199
        [0.7192118226600985, 0.7141424272818456, 0.7147087857847976, 0.7165048543689322] 0.7161419725239184
                     precision    recall  f1-score   support
                  0       1.00      1.00      1.00     56863
                  1       0.69      0.74      0.71       492
        avg / total       1.00      0.99      1.00     57355
        [0.724172517552658, 0.72210953346856, 0.7141424272818456, 0.7144259077526987] 0.7187125965139406
        """
        for epsilon in epsilons:
            f1_list = []
            for train_index, test_index in kf.split(data_set):
                # print("TRAIN:", train_index, "TEST:", test_index)
                train_set, cv_set = data_set[train_index], data_set[test_index]
                #1、参数估计
                mu = np.mean(train_set, axis=0)
                sigma = (train_set - mu).T.dot((train_set - mu))/train_set.shape[0]
                #2、求出联合概率分布

                pp_list, pf_list = [], []
                for x in range(cv_set.shape[0]):
                    pp_list.append(prob(cv_set[x]))
                print('正常样本的p值范围：{}-{}'.format(min(pp_list), max(pp_list)))  #
                for x in range(negtive_set.shape[0]):
                    pf_list.append(prob(negtive_set[x]))
                print('异常样本的p值范围：{}-{}'.format(min(pf_list), max(pf_list)))  #
                predictions = []
                for p in (pp_list+pf_list):
                    if p < epsilon:
                        predictions.append(1)
                    else:
                        predictions.append(0)
                print(classification_report([0]*cv_set.shape[0]+[1]*negtive_set.shape[0], predictions))
                f1 = f1_score([0]*cv_set.shape[0]+[1]*negtive_set.shape[0], predictions)
                f1_list.append(f1)
            print(f1_list, np.mean(f1_list))

    if mode == 2:
        """
        利用多元高斯分布求解异常监测1 假设特征间彼此独立
        https://www.bilibili.com/video/av9912938/?p=91
        
        步骤：
        1，参数估计：利用正常样本求出每个特征的均值ui和方差sigmai**2
        2，由特征间独立假设，得到联合概率密度分布，即每个特征分布的乘积
        3，利用网格法，确定最优阈值epsilon
            
        """
        for train_index, test_index in kf.split(data_set):
            # print("TRAIN:", train_index, "TEST:", test_index)
            train_set, cv_set = data_set[train_index], data_set[test_index]
            # 求每个特征的均值ui和方差sigmai**2
            ui_list, sigmai_list = [], []
            for i in range(train_set.shape[1]):
                ui_list.append(np.mean(train_set[:, i]))
                sigmai_list.append(np.std(train_set[:, i]))
            # 联合概率密度函数
            def prob(x):
                p = 1
                for i in range(train_set.shape[1]):
                    p = p * 1/(math.sqrt(2 * math.pi) * sigmai_list[i]) * math.exp(-(x[i]-ui_list[i])**2/(2*sigmai_list[i]**2))
                return p
            pp_list, pf_list = [], []
            for i in range(train_set.shape[0]):
                p = prob(train_set[i])
                pp_list.append(p)
            print('正常样本的p值范围：{}-{}'.format(min(pp_list), max(pp_list)))#0.0-3.8534293655740184e-14
            for i in range(test_set.shape[0]):
                p = prob(test_set[i])
                pf_list.append(p)
            print('异常样本的p值范围：{}-{}'.format(min(pf_list), max(pf_list)))#0.0-2.352825898167053e-15
            # picklew.dump2File(pp_list, 'pp_list.pkl')
            # picklew.dump2File(pf_list, 'pf_list.pkl')
            # pp_list = picklew.loadFromFile('pp_list.pkl')
            # pf_list = picklew.loadFromFile('pf_list.pkl')
            #网格法确定最优epsilon（即f1值最大的epsilon）
            def grid(epsilons):
                for epsilon in epsilons:
                    pps = pd.Series(pp_list)
                    pps[pps >= epsilon] = 1
                    pps[pps < epsilon] = 0
                    r0 = pps.value_counts()
                    # print(r0)
                    pfs = pd.Series(pf_list)
                    pfs[pfs >= epsilon] = 1
                    pfs[pfs < epsilon] = 0
                    r1 = pfs.value_counts()
                    # print(r1)
                    #计算异常样本的召回率、准确率、f1score
                    recall = r1[0]/(r1[0]+r1[1])
                    precision = r1[0]/(r1[0]+r0[0])
                    f1 = 2/(1/recall+1/precision)
                    print('准确率：{:.2%}--召回率：{:.2%}--f1：{:.2%}'.format(precision, recall, f1))
                    print(roc_auc_score(pd.Series([1]*284315+[0]*492), pps.append(pfs)))
                    # print(classification_report(pd.Series([1]*284315+[0]*492), pps.append(pfs)))
            # epsilons = [2.352825898167053e-15, 1e-15, 5e-16]
            """
            准确率：0.20%--召回率：99.80%--f1：0.41%
            准确率：0.22%--召回率：99.19%--f1：0.45%
            准确率：0.24%--召回率：98.98%--f1：0.48%      
            """
            epsilons = [1e-103, 1e-105, 1e-107]
            """
            准确率：15.44%--召回率：41.06%--f1：22.44%
            0.7033395270818508
            准确率：15.93%--召回率：40.85%--f1：22.92%
            0.702402404495529
            准确率：15.91%--召回率：39.63%--f1：22.70%  
            0.6963576019041058    
            
            最终 epsilon大约在1e-105附近取到最优解，f1：22.92%
            """
            grid(epsilons)
    if mode == 1:
        """
        data_explore
        """
        print(df.shape)#(284807, 31)
        # print(df['Class'].value_counts())
        """
        0    284315
        1       492
        """
        # print('ratio:{:.2f}'.format(284315/492))#ratio:577.88
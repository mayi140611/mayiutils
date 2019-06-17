#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: argparse_wrapper.py
@time: 2019-06-12 18:14

argparse是python的一个命令行解析包，非常编写可读性非常好的程序
"""
import argparse


class ArgparseWrapper:
    @classmethod
    def parseArgs(cls):

        parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
        parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
        parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
        parser.add_argument('--CRF', type=bool, default=True, help='use CRF at the top layer. if False, use Softmax')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        return parser.parse_args()


if __name__ == '__main__':
    args = ArgparseWrapper.parseArgs()
    # print(args)  # Namespace(CRF=True, batch_size=64, lr=0.001, train_data='data_path')
    """
    python argparse_wrapper.py --lr=999
    """
    print(args)  # Namespace(CRF=True, batch_size=64, lr=999.0, train_data='data_path')
    print(args.train_data)  # data_path


# coding=utf-8
import copy
from utils import utils
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict
from data_processors.DataProcessor import DataProcessor
from utils.global_p import *
import re


class ProLogicDP(DataProcessor):
    # data dict中存储模型所需特征信息的key（负例feed_dict需要append在最后）
    # The key in data dict to store the feature information needed by the model (feed_dict of negative examples should be appended at the end)
    data_columns = [X]
    info_columns = [SAMPLE_ID]

    @staticmethod
    def parse_dp_args(parser):
        """
        数据处理生成batch的命令行参数
        :param parser:
        :return:
        
        Command-line parameters to generate batches in data processing
        :param parser:
        :return:
        """
        parser.add_argument('--shuffle_or', type=int, default=1,
                            help='whether shuffle the or-list of each sent during training.')
        parser.add_argument('--shuffle_and', type=int, default=1,
                            help='whether shuffle the and-list of each sent during training.')
        return DataProcessor.parse_dp_args(parser)

    def __init__(self, shuffle_or, shuffle_and, *args, **kwargs):
        self.shuffle_or = shuffle_or
        self.shuffle_and = shuffle_and
        DataProcessor.__init__(self, *args, **kwargs)
        assert self.rank == 0

    def get_feed_dict(self, data, batch_start, batch_size, train, neg_data=None, special_cols=None):
        feed_dict = DataProcessor.get_feed_dict(
            self, data, batch_start, batch_size, train, neg_data=neg_data,
            special_cols=[X])
        x = [[i for i in s] for s in feed_dict[X]]
        if train:
            if self.shuffle_and:
                x = [[list(np.random.choice(o, size=len(o), replace=False)) for o in s] for s in x]
            if self.shuffle_or:
                for s in x:
                    np.random.shuffle(s)
        max_or_length = max([len(s) for s in x])
        for s in x:
            while len(s) < max_or_length:
                s.append([])
        max_and_length = max([max([len(s[i]) for s in x]) for i in range(max_or_length)])
        x = [[s[i] + [0] * (max_and_length - len(s[i])) for i in range(max_or_length)] for s in x]
        or_length = [len(i) for i in x[0]]
        feed_dict[X] = utils.numpy_to_torch(np.array(x), gpu=False)
        feed_dict[K_OR_LENGTH] = or_length
        # print(feed_dict)
        # assert 1==2
        return feed_dict

    def format_data_dict(self, df, model):
        data_loader = self.data_loader
        data = {}
        # label 记录在 'Y' 中
        # label is recorded in 'Y'
        if data_loader.label in df.columns:
            data[Y] = np.array(df[data_loader.label], dtype=np.float32)
        else:
            logging.warning('No Labels In Data: ' + data_loader.label)
            data['Y'] = np.zeros(len(df), dtype=np.float32)
        or_list = df[C_SENT].apply(lambda x: x.split('v'))
        x_list = or_list.apply(lambda x: [i.split('^') for i in x])
        x_list = x_list.apply(lambda x: [[int(v.replace('~', '-')) for v in i] for i in x])
        # or_list_length = or_list.apply(lambda x: [len(i.split('^')) for i in x])
        # elements = df[C_SENT].apply(lambda x: re.split('\^|v', x))
        # x_tag = elements.apply(lambda x: [0 if i.startswith('~') else 1 for i in x])
        # x = elements.apply(lambda x: [int(i[1:]) if i.startswith('~') else int(i) for i in x])
        data[X] = x_list.values
        return data

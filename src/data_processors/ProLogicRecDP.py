# coding=utf-8
from utils import utils
import numpy as np
import torch
from data_processors.DataProcessor import DataProcessor
from data_processors.HistoryDP import HistoryDP
from utils.global_p import *


class ProLogicRecDP(HistoryDP):
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
        parser.add_argument('--shuffle_his', type=int, default=0,
                            help='whether shuffle the his-list of each sent during training.')
        return HistoryDP.parse_dp_args(parser)

    def __init__(self, shuffle_his, *args, **kwargs):
        self.shuffle_his = shuffle_his
        HistoryDP.__init__(self, *args, **kwargs)
        assert self.sparse_his == 0

    def get_feed_dict(self, data, batch_start, batch_size, train, neg_data=None, special_cols=None):
        """
        topn模型产生一个batch，如果是训练需要对每个正样本采样一个负样本，保证每个batch前一半是正样本，后一半是对应的负样本
        :param data: data dict，由self.get_*_data()和self.format_data_dict()系列函数产生
        :param batch_start: batch开始的index
        :param batch_size: batch大小
        :param train: 训练还是测试
        :param neg_data: 负例的data dict，如果已经有可以传入拿来用
        :param special_cols: 需要特殊处理的column
        :return: batch的feed dict
        
        topn model will produce a batch, if doing training then need to sample a negative example for each positive example, and garanttee that for each batch the first half are positive examples and the second half are negative examples
        :param data: data dict, produced by self.get_*_data() and self.format_data_dict() functions
        :param batch_start: starting index of the batch
        :param batch_size: batch size
        :param train: training or testing
        :param neg_data: data dict of negative examples, if alreay exist can use directly
        :param special_cols: columns that need special treatment
        :return: feed dict of the batch
        """
        feed_dict = DataProcessor.get_feed_dict(
            self, data, batch_start, batch_size, train, neg_data=neg_data,
            special_cols=[C_HISTORY, C_HISTORY_NEG]
            if special_cols is None else [C_HISTORY, C_HISTORY_NEG] + special_cols)

        assert C_HISTORY_NEG not in feed_dict
        d = [[i for i in x] for x in feed_dict[C_HISTORY]]
        if train and self.shuffle_his:
            d = [list(np.random.choice(x, len(x), replace=False)) if len(x) != 0 else [] for x in d]

        lengths = [len(iids) for iids in d]
        max_length = max(lengths)
        new_d = np.array([x + [0] * (max_length - len(x)) for x in d])

        feed_dict[C_HISTORY] = utils.numpy_to_torch(new_d, gpu=False)
        feed_dict[C_HISTORY_LENGTH] = lengths
        return feed_dict

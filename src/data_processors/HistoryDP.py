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


class HistoryDP(DataProcessor):
    # data dict中存储模型所需特征信息的key，需要转换为tensor
    # The key to store the feature information needed by the model in data dict
    data_columns = [UID, IID, X, C_HISTORY, C_HISTORY_NEG]
    info_columns = [SAMPLE_ID, TIME, C_HISTORY_LENGTH, C_HISTORY_NEG_LENGTH]  # data dict中存储模额外信息的key

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
        # Some of these args are used by data_loader.append_his(all_his, max_his, neg_his, neg_column)
        parser.add_argument('--all_his', type=int, default=0,
                            help='Append all history in the training set')
        parser.add_argument('--max_his', type=int, default=-1,
                            help='Max history length. All his if max_his <= 0')
        parser.add_argument('--neg_his', type=int, default=0,
                            help='Whether keep negative interactions in the history')
        parser.add_argument('--neg_column', type=int, default=0,
                            help='Whether keep negative interactions in the history as a single column')
        parser.add_argument('--sparse_his', type=int, default=0,
                            help='Whether use sparse representation of user history.')
        parser.add_argument('--sup_his', type=int, default=0,
                            help='If sup_his > 0, supplement history list with 0')
        parser.add_argument('--drop_first', type=int, default=1,
                            help='If drop_first > 0, drop the first user interacted item with no previous history')
        return DataProcessor.parse_dp_args(parser)

    def __init__(self, max_his, sup_his, sparse_his, drop_first, *args, **kwargs):
        self.max_his = max_his
        self.sparse_his = sparse_his
        self.sup_his = sup_his
        self.drop_first = drop_first
        DataProcessor.__init__(self, *args, **kwargs)

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

        his_cs, his_ls = [C_HISTORY], [C_HISTORY_LENGTH]
        # 如果有负向历史的列
        # If there are columns of negative histories
        if C_HISTORY_NEG in feed_dict:
            his_cs.append(C_HISTORY_NEG)
            his_ls.append(C_HISTORY_NEG_LENGTH)

        for i, c in enumerate(his_cs):
            lc, d = his_ls[i], feed_dict[c]
            # 如果是稀疏表示
            # If it's sparse representation
            if self.sparse_his == 1:
                x, y, v = [], [], []
                for idx, iids in enumerate(d):
                    x.extend([idx] * len(iids))
                    y.extend([abs(iid) for iid in iids])
                    v.extend([1.0 if iid > 0 else -1.0 if iid < 0 else 0 for iid in iids])
                if len(x) <= 0:
                    i = utils.numpy_to_torch(np.array([[0], [0]]), gpu=False)
                    v = utils.numpy_to_torch(np.array([0.0], dtype=np.float32), gpu=False)
                else:
                    i = utils.numpy_to_torch(np.array([x, y]), gpu=False)
                    v = utils.numpy_to_torch(np.array(v, dtype=np.float32), gpu=False)
                history = torch.sparse.FloatTensor(
                    i, v, torch.Size([len(d), self.data_loader.item_num]))
                # if torch.cuda.device_count() > 0:
                #     history = history.cuda()
                feed_dict[c] = history
                feed_dict[lc] = [len(iids) for iids in d]
                # feed_dict[lc] = utils.numpy_to_torch(np.array([len(iids) for iids in d]), gpu=False)
            else:
                lengths = [len(iids) for iids in d]
                max_length = max(lengths)
                new_d = np.array([x + [0] * (max_length - len(x)) for x in d])
                feed_dict[c] = utils.numpy_to_torch(new_d, gpu=False)
                feed_dict[lc] = lengths
                # feed_dict[lc] = utils.numpy_to_torch(np.array(lengths), gpu=False)
        # print(feed_dict)
        # assert 1==2
        return feed_dict

    def format_data_dict(self, df, model):
        """
        除了常规的uid,iid,label,user、item、context特征外，还需处理历史交互
        :param df: 训练、验证、测试df
        :param model: Model类
        :return:
        
        Except for normal uid,iid,label,user, item, context features, also need to process history interactions
        :param df: training, validation, and testing df
        :param model: Model class
        :return:
        """
        # history需要data_loader放在df里
        # history need to be put into df by data_loader
        assert C_HISTORY in df

        his_cs = [C_HISTORY]
        if C_HISTORY_NEG in df:  # 如果有负向历史的列 (if there exist columns of negative histories)
            his_cs.append(C_HISTORY_NEG)
        if self.drop_first == 1:
            for c in his_cs:
                df = df[df[c].apply(lambda x: len(x) > 0)]
        data_dict = DataProcessor.format_data_dict(self, df, model)

        for c in his_cs:
            his = df[c].apply(lambda x: eval('[' + x + ']'))
            # if self.max_his > 0:
            #     his = his.apply(lambda x: x[:self.max_his])
            #     if self.sup_his == 1:
            #         his = his.apply(lambda x: x + [0] * (self.max_his - len(x)))
            data_dict[c] = his.values
        return data_dict

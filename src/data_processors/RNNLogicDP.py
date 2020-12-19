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


class RNNLogicDP(DataProcessor):
    # data dict中存储模型所需特征信息的key（负例feed_dict需要append在最后）
    # The key in data dict to store the feature information needed by the model (feed_dict of negative examples should be appended at the end)
    data_columns = [X]
    info_columns = [SAMPLE_ID]

    def __init__(self, *args, **kwargs):
        DataProcessor.__init__(self, *args, **kwargs)
        assert self.rank == 0

    def get_feed_dict(self, data, batch_start, batch_size, train, neg_data=None, special_cols=None):
        feed_dict = DataProcessor.get_feed_dict(
            self, data, batch_start, batch_size, train, neg_data=neg_data,
            special_cols=[X])

        lengths = [len(seq) for seq in feed_dict[X]]
        max_length = max(lengths)
        new_d = np.array([x + [0] * (max_length - len(x)) for x in feed_dict[X]])
        feed_dict[X] = utils.numpy_to_torch(new_d, gpu=False)
        feed_dict[K_S_LENGTH] = lengths
        # print(feed_dict)
        # assert 1 == 2
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
            data[Y] = np.zeros(len(df), dtype=np.float32)

        and_no = data_loader.variable_num
        or_no, not_no = and_no + 1, and_no + 2

        def sent_to_list(sent):
            result = sent.replace('^', ',%d,' % and_no)
            result = result.replace('v', ',%d,' % or_no)
            result = result.replace('~', '%d,' % not_no)
            result = result.split(',')
            result = [int(i) for i in result]
            return result

        elements = df[C_SENT].apply(lambda x: sent_to_list(x))
        data[X] = elements.values
        assert len(data[X]) == len(data[Y])
        return data

# coding=utf-8
import os
import pandas as pd
import numpy as np
from collections import Counter
import logging
import json
from data_loaders.DataLoader import DataLoader
from utils.global_p import *


class ProLogicDL(DataLoader):
    def __init__(self, path, dataset, *args, **kwargs):
        """
        初始化
        :param path: 数据集目录
        :param dataset: 数据集名称
        :param label: 标签column的名称
        :param load_data: 是否要载入数据文件，否则只载入数据集信息
        :param sep: csv的分隔符
        :param seqs_sep: 变长column的内部分隔符，比如历史记录可能为'1,2,4'
        
        Initialization
        :param path: path of the dataset
        :param dataset: name of the dataset
        :param label: label name of the columns
        :param load_data: if or not to load the dataset file, otherwise only load the dataset information
        :param sep: separator token of the csv
        :param seq_sep: internal separator token of variable-length columns, e.g., user history records could be '1,2,4'
        """
        self.variable_file = os.path.join(os.path.join(path, dataset), dataset + VARIABLE_SUFFIX)
        self.variable_df = pd.read_csv(self.variable_file, sep='\t')
        DataLoader.__init__(self, path=path, dataset=dataset, *args, **kwargs)

    def _load_info(self):
        """
        载入数据集信息文件，如果不存在则创建
        :return:
        
        Load the dataset information file, if non-existing then create
        :return:
        """
        DataLoader._load_info(self)
        self.column_max['variable'] = self.variable_df['variable'].max()
        self.column_min['variable'] = self.variable_df['variable'].min()
        self.variable_num = self.column_max['variable'] + 1
        logging.info("# of variables: %d" % self.variable_num)

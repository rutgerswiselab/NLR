# coding=utf-8

import torch
import torch.nn.functional as F
from models.DeepModel import DeepModel
from utils.global_p import *


class CNNLogic(DeepModel):
    include_id = False
    include_user_features = False
    include_item_features = False
    include_context_features = False
    data_loader = 'ProLogicDL'
    data_processor = 'RNNLogicDP'

    @staticmethod
    def parse_model_args(parser, model_name='CNNLogic'):
        parser.add_argument('--filter_size', type=str, default="[2,4,6,8]",
                            help='list or int, means the size of filters')
        parser.add_argument('--filter_num', type=str, default="[16,16,16,16]",
                            help='list or int, means the number of filters')
        parser.add_argument('--pooling', type=str, default="sum",
                            help='Pooling type: sum, min, max, mean')
        return DeepModel.parse_model_args(parser, model_name)

    def __init__(self, filter_size, filter_num, pooling, variable_num, feature_num=-1, *args, **kwargs):
        self.pooling = pooling.lower()
        self.filter_size = filter_size if type(filter_size) == list else eval(filter_size)
        self.filter_num = filter_num if type(filter_num) == list else eval(filter_num)
        if type(self.filter_size) is int:
            self.filter_size = [self.filter_size]
        if type(self.filter_num) is int:
            self.filter_num = [self.filter_num]
        assert len(self.filter_size) == len(self.filter_num)
        assert len(self.filter_size) > 0
        DeepModel.__init__(self, feature_num=variable_num + 3, *args, **kwargs)
        assert self.label_min == 0
        assert self.label_max == 1

    def _init_weights(self):
        self.feature_embeddings = torch.nn.Embedding(self.feature_num, self.f_vector_size)
        self.l2_embeddings = ['feature_embeddings']

        for i, size in enumerate(self.filter_size):
            setattr(self, 'conv_%d' % size, torch.nn.Conv2d(
                in_channels=1, out_channels=self.filter_num[i], kernel_size=(size, self.f_vector_size)))

        pre_size = sum(self.filter_num)
        for i, layer_size in enumerate(self.layers):
            setattr(self, 'layer_%d' % i, torch.nn.Linear(pre_size, layer_size))
            # setattr(self, 'bn_%d' % i, torch.nn.BatchNorm1d(layer_size))
            pre_size = layer_size
        self.prediction = torch.nn.Linear(pre_size, 1)

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        lengths = feed_dict[K_S_LENGTH]  # B
        sents = feed_dict[X]  # B * L
        valid_words = sents.gt(0).long()  # B * L
        sent_lengths = valid_words.sum(dim=-1)  # B

        sents_vectors = self.feature_embeddings(sents) * valid_words.unsqueeze(dim=-1).float()  # B * L * V
        sents_vectors = sents_vectors.unsqueeze(dim=1)  # B * 1 * L * V
        conv_vectors = []
        for i, size in enumerate(self.filter_size):
            conv2d = getattr(self, 'conv_%d' % size)(sents_vectors).squeeze(dim=-1)  # B * num * (L-size+1)
            if self.pooling == 'sum':
                conv_vector = conv2d.sum(dim=-1)  # B * num
            elif self.pooling == 'max':
                conv_vector = conv2d.max(dim=-1)  # B * num
            elif self.pooling == 'min':
                conv_vector = conv2d.min(dim=-1)  # B * num
            else:
                conv_vector = conv2d.mean(dim=-1)  # B * num
            conv_vectors.append(conv_vector)

        pre_layer = torch.cat(conv_vectors, dim=-1)  # B * sum(filter_num)
        for i in range(0, len(self.layers)):
            pre_layer = getattr(self, 'layer_%d' % i)(pre_layer)
            # pre_layer = getattr(self, 'bn_%d' % i)(pre_layer)
            pre_layer = F.relu(pre_layer)
            pre_layer = torch.nn.Dropout(p=feed_dict[DROPOUT])(pre_layer)
        prediction = self.prediction(pre_layer).sigmoid().view([-1])

        out_dict = {PREDICTION: prediction,
                    CHECK: check_list,
                    EMBEDDING_L2: embedding_l2}
        return out_dict

    def forward(self, feed_dict):
        """
        除了预测之外，还计算loss
        :param feed_dict: 型输入，是个dict
        :return: 输出，是个dict，prediction是预测值，check是需要检查的中间结果，loss是损失
        """
        out_dict = self.predict(feed_dict)
        check_list = out_dict[CHECK]
        prediction, label = out_dict[PREDICTION], feed_dict[Y]
        check_list.append(('prediction', prediction))
        check_list.append(('label', label))

        # loss = -(label * prediction.log()).sum() - ((1 - label) * (1 - prediction).log()).sum()
        if self.loss_sum == 1:
            loss = torch.nn.BCELoss(reduction='sum')(prediction, label)
        else:
            loss = torch.nn.MSELoss(reduction='mean')(prediction, label)
        # out_dict['loss'] = loss
        out_dict[LOSS] = loss
        out_dict[LOSS_L2] = self.l2(out_dict)
        out_dict[CHECK] = check_list
        return out_dict

# coding=utf-8

import torch
import torch.nn.functional as F
from models.DeepModel import DeepModel
from utils.global_p import *


class RNNLogic(DeepModel):
    include_id = False
    include_user_features = False
    include_item_features = False
    include_context_features = False
    data_loader = 'ProLogicDL'
    data_processor = 'RNNLogicDP'

    @staticmethod
    def parse_model_args(parser, model_name='RNNLogic'):
        parser.add_argument('--rnn_type', type=str, default="LSTM",
                            help='RNN/LSTM/GRU.')
        parser.add_argument('--rnn_bi', type=int, default=0,
                            help='1=bi-rnn/lstm/gru')
        return DeepModel.parse_model_args(parser, model_name)

    def __init__(self, rnn_type, rnn_bi, variable_num, feature_num=-1, *args, **kwargs):
        self.rnn_type = rnn_type.lower()
        self.rnn_bi = rnn_bi
        DeepModel.__init__(self, feature_num=variable_num + 3, *args, **kwargs)
        assert self.label_min == 0
        assert self.label_max == 1

    def _init_weights(self):
        self.feature_embeddings = torch.nn.Embedding(self.feature_num, self.f_vector_size)
        self.l2_embeddings = ['feature_embeddings']
        if self.rnn_type == 'gru':
            self.encoder = torch.nn.GRU(input_size=self.f_vector_size, hidden_size=self.f_vector_size,
                                        batch_first=True, bidirectional=self.rnn_bi == 1)
        elif self.rnn_type == 'lstm':
            self.encoder = torch.nn.LSTM(input_size=self.f_vector_size, hidden_size=self.f_vector_size,
                                         batch_first=True, bidirectional=self.rnn_bi == 1)
        else:
            self.encoder = torch.nn.RNN(input_size=self.f_vector_size, hidden_size=self.f_vector_size,
                                        batch_first=True, bidirectional=self.rnn_bi == 1)
        if self.rnn_bi == 1:
            pre_size = self.f_vector_size * 2
        else:
            pre_size = self.f_vector_size
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

        # Sort
        sort_sent_lengths, sort_idx = torch.topk(sent_lengths, k=len(lengths))  # B
        sort_sent_vectors = sents_vectors.index_select(dim=0, index=sort_idx)  # B * H * V

        # Pack
        sents_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_sent_vectors, sort_sent_lengths, batch_first=True)

        # RNN
        if self.rnn_type == 'lstm':
            output_rnn, (hidden_rnn, _) = self.encoder(sents_packed, None)
        else:
            output_rnn, hidden_rnn = self.encoder(sents_packed, None)
        if self.rnn_bi == 1:
            sort_pre_layer = torch.cat((hidden_rnn[0], hidden_rnn[1]), dim=-1)  # B * 2V
        else:
            sort_pre_layer = hidden_rnn[0]  # B * V

        # Unsort
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]  # B
        pre_layer = sort_pre_layer.index_select(dim=0, index=unsort_idx)  # B * V

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
        
        Except for making predictions, also compute the loss
        :param feed_dict: model input, it's a dict
        :return: output, it's a dict, prediction is the predicted value, check means needs to check the intermediate result, loss is the loss
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

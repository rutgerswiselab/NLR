# coding=utf-8

import torch
import torch.nn.functional as F
from models.GRU4Rec import GRU4Rec
from utils import utils
from utils.global_p import *


class NARM(GRU4Rec):
    data_processor = 'HistoryDP'  # Default data_processor

    @staticmethod
    def parse_model_args(parser, model_name='NARM'):
        parser.add_argument('--attention_size', type=int, default=16,
                            help='Size of attention hidden space.')
        return GRU4Rec.parse_model_args(parser, model_name)

    def __init__(self, attention_size, *args, **kwargs):
        self.attention_size = attention_size
        GRU4Rec.__init__(self, *args, **kwargs)

    def _init_weights(self):
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.l2_embeddings = ['iid_embeddings']
        if self.neg_emb == 1:
            self.iid_embeddings_neg = torch.nn.Embedding(self.item_num, self.ui_vector_size)
            self.l2_embeddings.append('iid_embeddings_neg')
        elif len(self.neg_layer) > 0:
            pre_size = self.ui_vector_size
            for i, layer_size in enumerate(self.neg_layer):
                setattr(self, 'neg_layer_%d' % i, torch.nn.Linear(pre_size, layer_size))
                pre_size = layer_size
            self.neg_layer_out = torch.nn.Linear(pre_size, self.ui_vector_size)

        self.encoder_g = torch.nn.GRU(input_size=self.ui_vector_size, hidden_size=self.hidden_size, batch_first=True,
                                      num_layers=self.num_layers)
        self.encoder_l = torch.nn.GRU(input_size=self.ui_vector_size, hidden_size=self.hidden_size, batch_first=True,
                                      num_layers=self.num_layers)
        self.A1 = torch.nn.Linear(self.hidden_size, self.attention_size, bias=False)
        self.A2 = torch.nn.Linear(self.hidden_size, self.attention_size, bias=False)
        self.attention_out = torch.nn.Linear(self.attention_size, 1, bias=False)
        self.out = torch.nn.Linear(2 * self.hidden_size, self.ui_vector_size, bias=False)

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        i_ids = feed_dict[IID]
        history = feed_dict[C_HISTORY]  # B * H
        lengths = feed_dict[C_HISTORY_LENGTH]  # B

        valid_his = history.abs().gt(0).long()  # B * H
        his_pos_neg = history.ge(0).unsqueeze(-1).float()  # B * H * 1
        his_length = valid_his.sum(dim=-1)  # B

        pos_his_vectors = self.iid_embeddings(history.abs() * valid_his) * \
                          valid_his.unsqueeze(dim=-1).float()  # B * H * V
        if self.neg_emb == 1:
            neg_his_vectors = self.iid_embeddings_neg(history.abs() * valid_his) * valid_his.unsqueeze(dim=-1).float()
            his_vectors = pos_his_vectors * his_pos_neg + (-his_pos_neg + 1) * neg_his_vectors  # B * H * V
        elif len(self.neg_layer) > 1:
            pre_layer = pos_his_vectors  # B * H * V
            for i in range(0, len(self.neg_layer)):
                pre_layer = getattr(self, 'neg_layer_%d' % i)(pre_layer)
                pre_layer = F.relu(pre_layer)
            neg_his_vectors = self.neg_layer_out(pre_layer)  # B * H * V
            his_vectors = pos_his_vectors * his_pos_neg + (-his_pos_neg + 1) * neg_his_vectors  # B * H * V
            his_vectors = his_vectors * valid_his.unsqueeze(dim=-1).float()  # B * H * V
        else:
            his_vectors = (his_pos_neg - 0.5) * 2 * pos_his_vectors  # B * H * V
        embedding_l2.append(his_vectors)

        # Sort and Pack
        sort_his_lengths, sort_idx = torch.topk(his_length, k=len(lengths))  # B
        sort_his_vectors = his_vectors.index_select(dim=0, index=sort_idx)  # B * H * V
        history_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_his_vectors, sort_his_lengths, batch_first=True)

        # Encoding Layer
        _, hidden_g = self.encoder_g(history_packed, None)
        output_l, hidden_l = self.encoder_l(history_packed, None)
        output_l, _ = torch.nn.utils.rnn.pad_packed_sequence(output_l, batch_first=True)  # B * H * V

        # Unsort
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]  # B
        output_l = output_l.index_select(dim=0, index=unsort_idx)  # B * H * V
        hidden_g = hidden_g[-1].index_select(dim=0, index=unsort_idx)  # B * V

        # Attention Layer
        attention_l = self.A2(output_l)  # B * H * A
        attention_g = self.A1(hidden_g)  # B * A
        attention_value = self.attention_out(torch.sigmoid(attention_g.unsqueeze(1) + attention_l))  # B * H * 1

        exp_att_v = attention_value.exp() * valid_his.unsqueeze(-1).float()  # B * H * 1
        exp_att_v_sum = exp_att_v.sum(dim=1, keepdim=True)  # B * H * 1
        attention_value = exp_att_v / exp_att_v_sum  # B * H * 1
        c_l = (attention_value * output_l).sum(1)  # B * V

        # Output Layer
        pred_vector = self.out(torch.cat((hidden_g, c_l), dim=1))  # B * V

        cf_i_vectors = self.iid_embeddings(i_ids)  # B * V
        embedding_l2.append(cf_i_vectors)

        prediction = (pred_vector * cf_i_vectors).sum(dim=-1).view([-1])
        check_list.append(('prediction', prediction))

        out_dict = {PREDICTION: prediction,
                    CHECK: check_list, EMBEDDING_L2: embedding_l2}
        return out_dict

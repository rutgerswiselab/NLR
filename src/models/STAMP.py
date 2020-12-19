# coding=utf-8

import torch
import torch.nn.functional as F
from models.GRU4Rec import GRU4Rec
from utils import utils
from utils.global_p import *


class STAMP(GRU4Rec):
    data_processor = 'HistoryDP'  # Default data_processor

    @staticmethod
    def parse_model_args(parser, model_name='STAMP'):
        parser.add_argument('--attention_size', type=int, default=64,
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

        self.attention_wxi = torch.nn.Linear(self.ui_vector_size, self.attention_size, bias=True)
        self.attention_wxt = torch.nn.Linear(self.ui_vector_size, self.attention_size, bias=True)
        self.attention_wms = torch.nn.Linear(self.ui_vector_size, self.attention_size, bias=True)
        self.attention_out = torch.nn.Linear(self.attention_size, 1, bias=False)

        self.mlp_a = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size, bias=True)
        self.mlp_b = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size, bias=True)

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

        # Prepare Vectors
        xt = his_vectors[range(len(lengths)), his_length - 1, :]  # B * V
        ms = his_vectors.sum(dim=1) / his_length.view([-1, 1]).float()  # B * V

        # Attention
        att_wxi_v = self.attention_wxi(his_vectors)  # B * H * V
        att_wxt_v = self.attention_wxt(xt).unsqueeze(dim=1)  # B * 1 * V
        att_wms_v = self.attention_wms(ms).unsqueeze(dim=1)  # B * 1 * V
        att_v = self.attention_out((att_wxi_v + att_wxt_v + att_wms_v).sigmoid())  # B * H * 1
        ma = (his_vectors * att_v * valid_his.unsqueeze(dim=-1).float()).sum(dim=1)  # B * V

        # Output Layer
        hs = self.mlp_a(ma).tanh()  # B * V
        ht = self.mlp_b(xt).tanh()  # B * V
        cf_i_vectors = self.iid_embeddings(i_ids)  # B * V
        embedding_l2.append(cf_i_vectors)

        pred_vector = hs * ht  # B * V
        prediction = (pred_vector * cf_i_vectors).sum(dim=-1).view([-1])
        check_list.append(('prediction', prediction))

        out_dict = {PREDICTION: prediction,
                    CHECK: check_list, EMBEDDING_L2: embedding_l2}
        return out_dict

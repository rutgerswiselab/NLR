# coding=utf-8

import torch
import torch.nn.functional as F
from models.RecModel import RecModel
from utils import utils, components
from utils.global_p import *


class NAIS(RecModel):
    data_processor = 'HistoryDP'  # 默认data_processor

    def _init_weights(self):
        self.iid_embeddings_his = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.l2_embeddings = ['iid_embeddings_his', 'iid_embeddings', 'user_bias', 'item_bias']

        self.att_k = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.att_q = torch.nn.Linear(self.ui_vector_size, 1, bias=False)

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []

        total_batch_size = feed_dict[TOTAL_BATCH_SIZE]
        real_batch_size = feed_dict[REAL_BATCH_SIZE]
        u_ids = feed_dict[UID]  # B
        i_ids = feed_dict[IID]  # B
        cf_i_vectors = self.iid_embeddings(i_ids)  # B * iv
        embedding_l2.append(cf_i_vectors)

        history = feed_dict[C_HISTORY]  # B * H

        # sum of history interactions
        valid_his = history.gt(0).long()  # B * H
        if feed_dict[TRAIN]:  # remove target item
            if_target_item = 1 - history.eq(i_ids.view([-1, 1])).long()  # B * H
            valid_his = if_target_item * valid_his  # B * H
        his_vector = self.iid_embeddings_his(history * valid_his) * valid_his.unsqueeze(dim=-1).float()  # B * H * iv
        embedding_l2.append(his_vector)

        # attention
        att_key = self.att_k(his_vector * cf_i_vectors.unsqueeze(dim=-2)).relu()  # B * H * iv
        att_query = self.att_q(att_key)  # B * H * 1
        his_vector = components.qk_attention(
            query=att_query, key=1, value=his_vector, valid=valid_his, beta=0.5)  # B * iv

        # bias
        u_bias = self.user_bias(u_ids).flatten()  # B
        i_bias = self.item_bias(i_ids).flatten()  # B
        embedding_l2.extend([u_bias, i_bias])

        prediction = (his_vector * cf_i_vectors).sum(dim=1).flatten()  # B
        prediction = prediction + u_bias + i_bias  # B
        check_list.append(('prediction', prediction))

        out_dict = {PREDICTION: prediction,
                    CHECK: check_list, EMBEDDING_L2: embedding_l2}
        return out_dict

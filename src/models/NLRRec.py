# coding=utf-8

import torch
import torch.nn.functional as F
import logging
from sklearn.metrics import *
import numpy as np
from models.NLR import NLR
from utils import utils
from utils.global_p import *


class NLRRec(NLR):
    include_id = True
    include_user_features = False
    include_item_features = False
    include_context_features = False
    data_loader = 'DataLoader'
    data_processor = 'ProLogicRecDP'

    @staticmethod
    def parse_model_args(parser, model_name='NLRRec'):
        parser.add_argument('--variable_num', type=int, default=-1,
                            help='Placeholder of variable_num')
        parser.add_argument('--seq_rec', type=int, default=1,
                            help='Whether keep the order of sequence.')
        parser.add_argument('--or_and', type=int, default=1,
                            help='Whether or-and or and-or.')
        return NLR.parse_model_args(parser, model_name)

    def __init__(self, or_and, seq_rec, item_num, variable_num=-1, *args, **kwargs):
        self.or_and = or_and
        self.seq_rec = seq_rec
        NLR.__init__(self, variable_num=item_num, *args, **kwargs)

    def uniform_size(self, vector1, vector2, train):
        if len(vector1.size()) < len(vector2.size()):
            vector1 = vector1.expand_as(vector2)
        else:
            vector2 = vector2.expand_as(vector1)
        if train:
            r12 = torch.Tensor(vector1.size()[:-1]).uniform_(0, 1).bernoulli()
            r12 = utils.tensor_to_gpu(r12).unsqueeze(-1)
            new_v1 = r12 * vector1 + (-r12 + 1) * vector2
            new_v2 = r12 * vector2 + (-r12 + 1) * vector1
            return new_v1, new_v2
        return vector1, vector2

    def predict_and_or(self, feed_dict):
        check_list, embedding_l2 = [], []
        train = feed_dict[TRAIN]
        seq_rec = self.seq_rec == 1
        total_batch_size = feed_dict[TOTAL_BATCH_SIZE]  # = B
        real_batch_size = feed_dict[REAL_BATCH_SIZE]  # = rB

        history = feed_dict[C_HISTORY]  # B * H
        history_length = feed_dict[C_HISTORY_LENGTH]  # B

        his_pos_neg = history.ge(0).float().unsqueeze(-1)  # B * H * 1
        his_valid = history.abs().gt(0).float()  # B * H

        elements = self.feature_embeddings(history.abs())  # B * H * V
        not_elements = self.logic_not(elements)  # B * H * V
        elements = his_pos_neg * elements + (-his_pos_neg + 1) * not_elements  # B * H * V
        elements = elements * his_valid.unsqueeze(-1)  # B * H * V

        constraint = [elements.view([total_batch_size, -1, self.v_vector_size])]  # B * H * V
        constraint_valid = [his_valid.view([total_batch_size, -1])]  # B * H

        if self.seq_rec == 0:
            # 随机打乱顺序计算
            all_as, all_avs = [], []
            for i in range(max(history_length)):
                all_as.append(elements[:, i, :])  # B * V
                all_avs.append(his_valid[:, i].unsqueeze(-1))  # B * 1
            while len(all_as) > 1:
                idx_a, idx_b = 0, 1
                if train:
                    idx_a, idx_b = np.random.choice(len(all_as), size=2, replace=False)
                if idx_a > idx_b:
                    a, av = all_as.pop(idx_a), all_avs.pop(idx_a)  # B * V,  B * 1
                    b, bv = all_as.pop(idx_b), all_avs.pop(idx_b)  # B * V,  B * 1
                else:
                    b, bv = all_as.pop(idx_b), all_avs.pop(idx_b)  # B * V,  B * 1
                    a, av = all_as.pop(idx_a), all_avs.pop(idx_a)  # B * V,  B * 1
                a_and_b = self.logic_and(a, b, train=train & ~seq_rec)  # B * V
                abv = av * bv  # B * 1
                ab = abv * a_and_b + av * (-bv + 1) * a + (-av + 1) * bv * b  # B * V
                all_as.insert(0, ab)
                all_avs.insert(0, (av + bv).gt(0).float())
                constraint.append(ab.view([total_batch_size, 1, self.v_vector_size]))
                constraint_valid.append(abv)
            and_vector = all_as[0]
            left_valid = all_avs[0]
        else:
            # # 按顺序计算
            tmp_a = None
            for i in range(max(history_length)):
                tmp_a_valid = his_valid[:, i].unsqueeze(-1)  # B * 1
                if tmp_a is None:
                    tmp_a = elements[:, i, :] * tmp_a_valid  # B * V
                else:
                    tmp_a = self.logic_and(tmp_a, elements[:, i, :], train=train & ~seq_rec) * tmp_a_valid + \
                            tmp_a * (-tmp_a_valid + 1)  # B * V
                    constraint.append(tmp_a.view([total_batch_size, 1, self.v_vector_size]))  # B * 1 * V
                    constraint_valid.append(tmp_a_valid)  # B * 1
            and_vector = tmp_a  # B * V
            left_valid = his_valid[:, 0].unsqueeze(-1)  # B * 1

        all_valid = utils.tensor_to_gpu(torch.ones([total_batch_size, 1]))  # B * 1

        left_vector = self.logic_not(and_vector)  # B * V
        constraint.append(left_vector.view([total_batch_size, 1, self.v_vector_size]))  # B * 1 * V
        constraint_valid.append(left_valid)  # B * 1

        right_vector = self.feature_embeddings(feed_dict[IID])  # B * V
        constraint.append(right_vector.view([total_batch_size, 1, self.v_vector_size]))  # B * 1 * V
        constraint_valid.append(all_valid)  # B * 1

        sent_vector = self.logic_or(left_vector, right_vector, train=train & ~seq_rec) * left_valid \
                      + (-left_valid + 1) * right_vector  # B * V
        # sent_vector = self.logic_or(left_vector, right_vector, train=train & ~seq_rec)  # B * V
        constraint.append(sent_vector.view([total_batch_size, 1, self.v_vector_size]))  # B * 1 * V
        constraint_valid.append(left_valid)  # B * 1

        if feed_dict[RANK] == 1:
            prediction = self.similarity(sent_vector, self.true, sigmoid=False).view([-1])
        else:
            prediction = self.similarity(sent_vector, self.true, sigmoid=True) * \
                         (self.label_max - self.label_min) + self.label_min

        check_list.append(('prediction', prediction))
        check_list.append(('label', feed_dict[Y]))
        check_list.append(('true', self.true))

        constraint = torch.cat(tuple(constraint), dim=1)
        constraint_valid = torch.cat(tuple(constraint_valid), dim=1)
        out_dict = {PREDICTION: prediction,
                    CHECK: check_list,
                    'constraint': constraint,
                    'constraint_valid': constraint_valid,
                    EMBEDDING_L2: embedding_l2}
        return out_dict

    def predict_or_and(self, feed_dict):
        check_list, embedding_l2 = [], []
        train = feed_dict[TRAIN]
        seq_rec = self.seq_rec == 0
        total_batch_size = feed_dict[TOTAL_BATCH_SIZE]  # = B
        real_batch_size = feed_dict[REAL_BATCH_SIZE]  # = rB

        history = feed_dict[C_HISTORY]  # B * H
        history_length = feed_dict[C_HISTORY_LENGTH]  # B

        his_pos_neg = history.ge(0).float().unsqueeze(-1)  # B * H * 1
        his_valid = history.abs().gt(0).float()  # B * H

        elements = self.feature_embeddings(history.abs())  # B * H * V
        embedding_l2.append(elements)
        constraint = [elements.view([total_batch_size, -1, self.v_vector_size])]  # B * H * V
        constraint_valid = [his_valid.view([total_batch_size, -1])]  # B * H

        not_elements = self.logic_not(elements)  # B * H * V
        constraint.append(not_elements.view([total_batch_size, -1, self.v_vector_size]))  # B * H * V
        constraint_valid.append(his_valid * (-his_pos_neg.view([total_batch_size, -1]) + 1))  # B * H

        elements = his_pos_neg * elements + (-his_pos_neg + 1) * not_elements  # B * H * V
        elements = elements * his_valid.unsqueeze(-1)  # B * H * V

        # # 随机打乱顺序计算
        if self.seq_rec == 0:
            all_os, all_ovs = [], []
            for i in range(max(history_length)):
                all_os.append(elements[:, i, :])  # B * V
                all_ovs.append(his_valid[:, i].unsqueeze(-1))  # B * 1
            while len(all_os) > 1:
                idx_a, idx_b = 0, 1
                if train:
                    idx_a, idx_b = np.random.choice(len(all_os), size=2, replace=False)
                if idx_a > idx_b:
                    a, av = all_os.pop(idx_a), all_ovs.pop(idx_a)  # B * V,  B * 1
                    b, bv = all_os.pop(idx_b), all_ovs.pop(idx_b)  # B * V,  B * 1
                else:
                    b, bv = all_os.pop(idx_b), all_ovs.pop(idx_b)  # B * V,  B * 1
                    a, av = all_os.pop(idx_a), all_ovs.pop(idx_a)  # B * V,  B * 1
                a_or_b = self.logic_or(a, b, train=train & ~seq_rec)  # B * V
                abv = av * bv  # B * 1
                ab = abv * a_or_b + av * (-bv + 1) * a + (-av + 1) * bv * b  # B * V
                all_os.insert(0, ab)
                all_ovs.insert(0, (av + bv).gt(0).float())
                constraint.append(ab.view([total_batch_size, 1, self.v_vector_size]))
                constraint_valid.append(abv)
            or_vector = all_os[0]
            left_valid = all_ovs[0]
        else:
            # # 按顺序计算o
            tmp_o = None
            for i in range(max(history_length)):
                tmp_o_valid = his_valid[:, i].unsqueeze(-1)  # B * 1
                if tmp_o is None:
                    tmp_o = elements[:, i, :] * tmp_o_valid  # B * V
                else:
                    tmp_o = self.logic_or(tmp_o, elements[:, i, :], train=train & ~seq_rec) * tmp_o_valid + \
                            tmp_o * (-tmp_o_valid + 1)  # B * V
                    constraint.append(tmp_o.view([total_batch_size, 1, self.v_vector_size]))  # B * 1 * V
                    constraint_valid.append(tmp_o_valid)  # B * 1
            or_vector = tmp_o  # B * V
            left_valid = his_valid[:, 0].unsqueeze(-1)  # B * 1

        all_valid = feed_dict[IID].gt(0).float().view([total_batch_size, 1])  # B * 1

        right_vector = self.feature_embeddings(feed_dict[IID])  # B * V
        embedding_l2.append(right_vector)
        constraint.append(right_vector.view([total_batch_size, 1, self.v_vector_size]))  # B * 1 * V
        constraint_valid.append(all_valid)  # B * 1

        sent_vector = self.logic_and(or_vector, right_vector, train=train & ~seq_rec) * left_valid \
                      + (-left_valid + 1) * right_vector  # B * V
        constraint.append(sent_vector.view([total_batch_size, 1, self.v_vector_size]))  # B * 1 * V
        constraint_valid.append(left_valid)  # B * 1

        if feed_dict[RANK] == 1:
            prediction = self.similarity(sent_vector, self.true, sigmoid=False).view([-1])
        else:
            prediction = self.similarity(sent_vector, self.true, sigmoid=True) * \
                         (self.label_max - self.label_min) + self.label_min

        check_list.append(('prediction', prediction))
        check_list.append(('label', feed_dict[Y]))
        check_list.append(('true', self.true))

        constraint = torch.cat(tuple(constraint), dim=1)
        constraint_valid = torch.cat(tuple(constraint_valid), dim=1)
        out_dict = {PREDICTION: prediction,
                    CHECK: check_list,
                    'constraint': constraint,
                    'constraint_valid': constraint_valid,
                    EMBEDDING_L2: embedding_l2}

        every_his = self.logic_and(elements, right_vector.view([total_batch_size, 1, self.v_vector_size]), train=False)
        every_his = self.similarity(every_his, self.true, sigmoid=True)
        target_sim = self.similarity(right_vector, self.true, sigmoid=True)
        every_his = torch.cat([every_his.view([total_batch_size, -1]),
                               target_sim.view([total_batch_size, -1])], dim=1)
        out_dict['sth'] = every_his
        return out_dict

    def predict(self, feed_dict):
        if self.or_and == 1:
            return self.predict_or_and(feed_dict)
        return self.predict_and_or(feed_dict)

    def forward(self, feed_dict):
        """
        除了预测之外，还计算loss
        :param feed_dict: 型输入，是个dict
        :return: 输出，是个dict，prediction是预测值，check是需要检查的中间结果，loss是损失
        """
        out_dict = self.predict(feed_dict)
        out_dict = self.logic_regularizer(out_dict, train=feed_dict[TRAIN])
        prediction, label = out_dict[PREDICTION], feed_dict[Y]
        r_loss = out_dict['r_loss']
        check_list = out_dict[CHECK]

        # loss
        if feed_dict[RANK] == 1:
            # 计算topn推荐的loss，batch前一半是正例，后一半是负例
            loss = self.rank_loss(out_dict[PREDICTION], feed_dict[Y], feed_dict[REAL_BATCH_SIZE])
        else:
            # 计算rating/clicking预测的loss，默认使用mse
            if self.loss_sum == 1:
                loss = torch.nn.MSELoss(reduction='sum')(out_dict[PREDICTION], feed_dict[Y])
            else:
                loss = torch.nn.MSELoss(reduction='mean')(out_dict[PREDICTION], feed_dict[Y])
        out_dict[LOSS] = loss + r_loss
        out_dict[LOSS_L2] = self.l2(out_dict)
        out_dict[CHECK] = check_list
        return out_dict

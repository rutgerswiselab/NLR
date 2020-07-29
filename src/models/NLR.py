# coding=utf-8

import torch
import torch.nn.functional as F
import logging
from sklearn.metrics import *
import numpy as np
from models.BaseModel import BaseModel
from utils import utils
from utils.global_p import *


class NLR(BaseModel):
    include_id = False
    include_user_features = False
    include_item_features = False
    include_context_features = False
    data_loader = 'ProLogicDL'
    data_processor = 'ProLogicDP'

    @staticmethod
    def parse_model_args(parser, model_name='NLR'):
        parser.add_argument('--v_vector_size', type=int, default=64,
                            help='Size of feature vectors.')
        parser.add_argument('--r_logic', type=float, default=0.1,
                            help='Weight of logic regularizer loss')
        parser.add_argument('--r_length', type=float, default=0.001,
                            help='Weight of vector length regularizer loss')
        parser.add_argument('--sim_scale', type=int, default=10,
                            help='Expand the raw similarity *sim_scale before sigmoid.')
        parser.add_argument('--sim_alpha', type=float, default=0,
                            help='Similarity function divide (square sum then sim_alpha)')
        parser.add_argument('--layers', type=int, default=1,
                            help='Number of or/and/not hidden layers.')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, variable_num, v_vector_size, layers, r_logic, r_length, sim_scale, sim_alpha,
                 *args, **kwargs):
        self.variable_num = variable_num
        self.v_vector_size = v_vector_size
        self.r_logic = r_logic
        self.r_length = r_length
        self.layers = layers
        BaseModel.__init__(self, *args, **kwargs)
        assert self.label_min == 0
        assert self.label_max == 1
        self.sim_scale = sim_scale
        self.sim_alpha = sim_alpha

    def _init_weights(self):
        self.feature_embeddings = torch.nn.Embedding(self.variable_num, self.v_vector_size)
        self.l2_embeddings = ['feature_embeddings']

        self.true = torch.nn.Parameter(utils.numpy_to_torch(
            np.random.uniform(0, 1, size=[1, self.v_vector_size]).astype(np.float32)), requires_grad=False)

        self.not_layer = torch.nn.Linear(self.v_vector_size, self.v_vector_size)
        for i in range(self.layers):
            setattr(self, 'not_layer_%d' % i, torch.nn.Linear(self.v_vector_size, self.v_vector_size))

        self.and_layer = torch.nn.Linear(self.v_vector_size * 2, self.v_vector_size)
        for i in range(self.layers):
            setattr(self, 'and_layer_%d' % i, torch.nn.Linear(self.v_vector_size * 2, self.v_vector_size * 2))

        self.or_layer = torch.nn.Linear(self.v_vector_size * 2, self.v_vector_size)
        for i in range(self.layers):
            setattr(self, 'or_layer_%d' % i, torch.nn.Linear(self.v_vector_size * 2, self.v_vector_size * 2))

        self.sim_layer = torch.nn.Linear(self.v_vector_size, 1)
        for i in range(self.layers):
            setattr(self, 'sim_layer_%d' % i, torch.nn.Linear(self.v_vector_size, self.v_vector_size))

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

    def logic_not(self, vector):
        for i in range(self.layers):
            vector = F.relu(getattr(self, 'not_layer_%d' % i)(vector))
        vector = self.not_layer(vector)
        return vector

    def logic_and(self, vector1, vector2, train):
        vector1, vector2 = self.uniform_size(vector1, vector2, train)
        vector = torch.cat((vector1, vector2), dim=-1)
        for i in range(self.layers):
            vector = F.relu(getattr(self, 'and_layer_%d' % i)(vector))
        vector = self.and_layer(vector)
        return vector

    def logic_or(self, vector1, vector2, train):
        vector1, vector2 = self.uniform_size(vector1, vector2, train)
        vector = torch.cat((vector1, vector2), dim=-1)
        for i in range(self.layers):
            vector = F.relu(getattr(self, 'or_layer_%d' % i)(vector))
        vector = self.or_layer(vector)
        return vector

    def mse(self, vector1, vector2):
        vector1, vector2 = self.uniform_size(vector1, vector2, train=False)
        return (vector1 - vector2) ** 2

    def dot_product(self, vector1, vector2):
        vector1, vector2 = self.uniform_size(vector1, vector2, train=False)
        result = (vector1 * vector2).sum(dim=-1)
        vector1_pow = vector1.pow(2).sum(dim=-1).pow(self.sim_alpha)
        vector2_pow = vector2.pow(2).sum(dim=-1).pow(self.sim_alpha)
        result = result / torch.clamp(vector1_pow * vector2_pow, min=1e-8)
        return result

    def similarity(self, vector1, vector2, sigmoid=True):
        result = F.cosine_similarity(vector1, vector2, dim=-1)
        result = result * self.sim_scale
        if sigmoid:
            return result.sigmoid()
        return result

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        train = feed_dict[TRAIN]
        batch_size = feed_dict[TOTAL_BATCH_SIZE]  # = B
        or_length = feed_dict[K_OR_LENGTH]  # O

        x = feed_dict[X]  # B * O * A
        x_pos_neg = torch.ge(x, 0).float().unsqueeze(-1)  # B * O * A * 1
        x_valid = torch.gt(torch.abs(x), 0).float()  # B * O * A

        elements = self.feature_embeddings(torch.abs(x))  # B * O * A * V
        not_elements = self.logic_not(elements)  # B * O * A * V
        elements = x_pos_neg * elements + (-x_pos_neg + 1) * not_elements  # B * O * A * V
        elements = elements * x_valid.unsqueeze(-1)  # B * O * A * V

        constraint = [elements.view([batch_size, -1, self.v_vector_size])]  # B * ? * V
        constraint_valid = [x_valid.view([batch_size, -1])]  # B * ?

        # # 随机打乱顺序计算
        all_os, all_ovs = [], []
        for o in range(len(or_length)):
            all_as, all_avs = [], []
            for a in range(or_length[o]):
                all_as.append(elements[:, o, a, :])  # B * V
                all_avs.append(x_valid[:, o, a].unsqueeze(-1))  # B * 1
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
                a_and_b = self.logic_and(a, b, train=train)  # B * V
                abv = av * bv  # B * 1
                ab = abv * a_and_b + av * (-bv + 1) * a + (-av + 1) * bv * b  # B * V
                all_as.insert(0, ab)
                all_avs.insert(0, (av + bv).gt(0).float())
                constraint.append(ab.view([batch_size, 1, self.v_vector_size]))
                constraint_valid.append(abv)
            all_os.append(all_as[0])
            all_ovs.append(all_avs[0])

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
            a_or_b = self.logic_or(a, b, train=train)  # B * V
            abv = av * bv  # B * 1
            ab = abv * a_or_b + av * (-bv + 1) * a + (-av + 1) * bv * b  # B * V
            all_os.insert(0, ab)
            all_ovs.insert(0, (av + bv).gt(0).float())
            constraint.append(ab.view([batch_size, 1, self.v_vector_size]))
            constraint_valid.append(abv)
        result_vector = all_os[0]  # B * V

        prediction = self.similarity(result_vector, self.true).view([-1])

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

    def logic_regularizer(self, out_dict, train):
        check_list = out_dict[CHECK]
        constraint = out_dict['constraint']
        constraint_valid = out_dict['constraint_valid']
        false = self.logic_not(self.true)

        # # regularizer
        # length
        r_length = constraint.norm(dim=2).sum()
        check_list.append(('r_length', r_length))

        # not
        r_not_true = self.similarity(false, self.true)
        r_not_true = r_not_true.sum()
        check_list.append(('r_not_true', r_not_true))
        r_not_self = self.similarity(self.logic_not(constraint), constraint)
        r_not_self = (r_not_self * constraint_valid).sum()
        check_list.append(('r_not_self', r_not_self))
        r_not_not_self = 1 - self.similarity(self.logic_not(self.logic_not(constraint)), constraint)
        r_not_not_self = (r_not_not_self * constraint_valid).sum()
        check_list.append(('r_not_not_self', r_not_not_self))

        # and
        r_and_true = 1 - self.similarity(self.logic_and(constraint, self.true, train=train), constraint)
        r_and_true = (r_and_true * constraint_valid).sum()
        check_list.append(('r_and_true', r_and_true))

        r_and_false = 1 - self.similarity(self.logic_and(constraint, false, train=train), false)
        r_and_false = (r_and_false * constraint_valid).sum()
        check_list.append(('r_and_false', r_and_false))

        r_and_self = 1 - self.similarity(self.logic_and(constraint, constraint, train=train), constraint)
        r_and_self = (r_and_self * constraint_valid).sum()
        check_list.append(('r_and_self', r_and_self))

        r_and_not_self = 1 - self.similarity(self.logic_and(constraint, self.logic_not(constraint), train=train), false)
        r_and_not_self = (r_and_not_self * constraint_valid).sum()
        check_list.append(('r_and_not_self', r_and_not_self))

        # or
        r_or_true = 1 - self.similarity(self.logic_or(constraint, self.true, train=train), self.true)
        r_or_true = (r_or_true * constraint_valid).sum()
        check_list.append(('r_or_true', r_or_true))

        r_or_false = 1 - self.similarity(self.logic_or(constraint, false, train=train), constraint)
        r_or_false = (r_or_false * constraint_valid).sum()
        check_list.append(('r_or_false', r_or_false))

        r_or_self = 1 - self.similarity(self.logic_or(constraint, constraint, train=train), constraint)
        r_or_self = (r_or_self * constraint_valid).sum()
        check_list.append(('r_or_self', r_or_self))

        r_or_not_self = 1 - self.similarity(self.logic_or(constraint, self.logic_not(constraint), train=train),
                                            self.true)
        r_or_not_self = (r_or_not_self * constraint_valid).sum()
        check_list.append(('r_or_not_self', r_or_not_self))

        r_loss = 0
        r_loss += r_not_true + r_not_self + r_not_not_self + \
                  r_and_true + r_and_false + r_and_self + r_and_not_self + \
                  r_or_true + r_or_false + r_or_self + r_or_not_self

        if self.r_logic > 0:
            r_loss = r_loss * self.r_logic
        else:
            r_loss = utils.numpy_to_torch(np.array(0.0, dtype=np.float32))
        r_loss += r_length * self.r_length
        check_list.append(('r_loss', r_loss))

        out_dict['r_loss'] = r_loss
        return out_dict

    def forward(self, feed_dict):
        """
        除了预测之外，还计算loss
        :param feed_dict: 型输入，是个dict
        :return: 输出，是个dict，prediction是预测值，check是需要检查的中间结果，loss是损失
        """
        out_dict = self.predict(feed_dict)
        out_dict = self.logic_regularizer(out_dict, feed_dict[TRAIN])
        prediction, label = out_dict[PREDICTION], feed_dict[Y]
        r_loss = out_dict['r_loss']
        check_list = out_dict[CHECK]

        if self.loss_sum == 1:
            loss = torch.nn.BCELoss(reduction='sum')(prediction, label)
        else:
            loss = torch.nn.MSELoss(reduction='mean')(prediction, label)

        out_dict[LOSS] = loss + r_loss
        out_dict[LOSS_L2] = self.l2(out_dict)
        out_dict[CHECK] = check_list
        return out_dict

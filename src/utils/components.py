# coding=utf-8

import torch
import torch.nn.functional as F

from utils import utils


def qk_attention(query, key, value, valid=None, beta=1):
    """
    :param query: ? * l * a
    :param key: ? * l * a
    :param value: ? * l * v
    :param valid: ? * l
    :param beta: smooth softmax
    :return: ? * v
    """
    ele_valid = 1 if valid is None else valid.unsqueeze(dim=-1)  # ? * l * 1
    att_v = (query * key).sum(dim=-1, keepdim=True)  # ? * l * 1
    att_exp = (att_v - att_v.max(dim=-2, keepdim=True)[0]).exp() * ele_valid.float()  # ? * l * 1
    att_sum = att_exp.sum(dim=-2, keepdim=True)  # ? * 1 * 1
    sum_valid = 1 if valid is None else ele_valid.sum(dim=-2, keepdim=True).gt(0).float()  # ? * 1 * 1
    att_w = att_exp / (att_sum * sum_valid + 1 - sum_valid).pow(beta)  # ? * l * 1
    result = (att_w * value).sum(dim=-2)  # ? * v
    return result


def seq_rnn(seq_vectors, valid, rnn, lstm=False, init=None):
    '''
    :param seq_vectors: b * l * v
    :param valid: b * l
    :param rnn: pytorch RNN object
    :param lstm:
    :param init:
    :return:
    '''
    seq_lengths = valid.sum(dim=-1)  # b
    n_samples = seq_lengths.size()[0]
    seq_lengths_valid = seq_lengths.gt(0).float().unsqueeze(dim=0).unsqueeze(dim=-1)  # 1 * b * 1
    seq_lengths_clamped = seq_lengths.clamp(min=1)  # b

    # Sort
    sort_seq_lengths, sort_idx = torch.topk(seq_lengths_clamped, k=n_samples)  # b
    sort_seq_vectors = seq_vectors.index_select(dim=0, index=sort_idx)  # b * l * v

    # Pack
    seq_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_seq_vectors, sort_seq_lengths, batch_first=True)

    # RNN
    if lstm:
        if init is not None:
            init = [i.index_select(dim=1, index=sort_idx) if i is not None else i for i in init]
        sort_output, (sort_hidden, _) = rnn(seq_packed, init)
    else:
        if init is not None:
            init = init.index_select(dim=1, index=sort_idx)
        sort_output, sort_hidden = rnn(seq_packed, init)
    sort_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
        sort_output, batch_first=True, total_length=valid.size()[1])  # b * l

    # Unsort
    unsort_idx = torch.topk(sort_idx, k=n_samples, largest=False)[1]  # b
    output = sort_output.index_select(dim=0, index=unsort_idx) * valid.unsqueeze(dim=-1).float()  # b * l * v/2v
    hidden = sort_hidden.index_select(dim=1, index=unsort_idx) * seq_lengths_valid  # 1/2 * b * v
    return output, hidden


def rank_loss(prediction, label, real_batch_size, loss_sum):
    '''
    计算rank loss，类似BPR-max，参考论文:
    @inproceedings{hidasi2018recurrent,
      title={Recurrent neural networks with top-k gains for session-based recommendations},
      author={Hidasi, Bal{\'a}zs and Karatzoglou, Alexandros},
      booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
      pages={843--852},
      year={2018},
      organization={ACM}
    }
    :param prediction: 预测值 [None]
    :param label: 标签 [None]
    :param real_batch_size: 观测值batch大小，不包括sample
    :param loss_sum: 1=sum, other= mean
    :return:
    
    Compute rank loss，similar to BPR-max，see reference:
    @inproceedings{hidasi2018recurrent,
      title={Recurrent neural networks with top-k gains for session-based recommendations},
      author={Hidasi, Bal{\'a}zs and Karatzoglou, Alexandros},
      booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
      pages={843--852},
      year={2018},
      organization={ACM}
    }
    :param prediction: predicted value [None]
    :param label: label [None]
    :param real_batch_size: batch size of observation data, excluding sample
    :param loss_sum: 1=sum, other= mean
    :return:
    '''
    pos_neg_tag = (label - 0.5) * 2
    observed, sample = prediction[:real_batch_size], prediction[real_batch_size:]
    # sample = sample.view([-1, real_batch_size]).mean(dim=0)
    sample = sample.view([-1, real_batch_size]) * pos_neg_tag.view([1, real_batch_size])  # ? * b
    sample_softmax = (sample - sample.max(dim=0)[0]).softmax(dim=0)  # ? * b
    sample = (sample * sample_softmax).sum(dim=0)  # b
    # loss = -(pos_neg_tag * (observed - sample)).sigmoid().log()
    loss = F.softplus(-pos_neg_tag * (observed - sample))
    if loss_sum == 1:
        return loss.sum()
    return loss.mean()


def cold_sampling(vectors, cs_ratio):
    """
    :param vectors: ? * v
    :param cs_ratio: 0 < cs_ratio < 1
    :return:
    """
    cs_p = torch.empty(vectors.size()[:-1]).fill_(cs_ratio).unsqueeze(dim=-1)  # ? * 1
    drop_pos = utils.tensor_to_gpu(torch.bernoulli(cs_p))  # ? * 1
    random_vectors = utils.tensor_to_gpu(torch.empty(vectors.size()).normal_(0, 0.01))  # ? * v
    cs_vectors = random_vectors * drop_pos + vectors * (1 - drop_pos)  # ? * v
    return cs_vectors

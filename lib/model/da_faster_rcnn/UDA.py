import torch
import torch.nn as  nn
import  numpy as np


def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)


def get_source_share_weight(domain_out, before_softmax):
    # before_softmax = before_softmax / class_temperature
    # after_softmax = nn.Softmax(-1)(before_softmax)
    # print("after_softmax的维度",after_softmax.shape)
    after_softmax=before_softmax

    domain_logit = reverse_sigmoid(domain_out)
    domain_logit = domain_logit / 1.0
    domain_out = nn.Sigmoid()(domain_logit)

    entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
    # print("类别个数",after_softmax.size(1))
    entropy_norm = entropy / np.log(after_softmax.size(1))

    weight = entropy_norm - domain_out
    weight = weight.detach()
    return weight


def get_target_share_weight(domain_out, before_softmax):
    return - get_source_share_weight(domain_out, before_softmax)


def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    x = x / torch.mean(x)
    return x.detach()


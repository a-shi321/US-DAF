from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time
'''
RPN网络结构就是：
生成anchors -> softmax分类器提取positvie anchors -> bbox reg回归positive anchors -> Proposal Layer生成proposals
'''
class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES  #[4,8,16,32]
        self.anchor_ratios = cfg.ANCHOR_RATIOS  #[0.5,1,2]
        self.feat_stride = cfg.FEAT_STRIDE[0]  #16

        # 定义conv层处理输入特征映射
        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        # 定义前景/背景分类层
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        # #4*3*2=24应该是这里变成12种anchors   #用来判别究竟是前景还是背景，用于存储目标分数
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # 定义锚框偏移量预测层
        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        #这里是48,4*3*4  #用于存储坐标分数
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer# 定义区域生成模块
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer# 定义生成RPN训练标签模块，仅在训练时使用
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)
        # RPN分类损失以及回归损失，仅在训练时计算
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):

        batch_size = base_feat.size(0)

        # return feature map after conv relu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        #print("rpn_conv1的维度", rpn_conv1.shape)
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)   #用来判别究竟是前景还是背景，用于存储目标分数torch.Size([1, 24, 37, 75])
        #print("rpn_cls_score的维度", rpn_cls_score.shape)
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        #print("rpn_cls_score_reshape的维度",rpn_cls_score_reshape.shape)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        #get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)    #用于存储坐标分数  torch.Size([1, 48, 37, 75])

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data, im_info, cfg_key))
        #print("ROIs",rois.shape)#源域数据输出2000个rois[1, 2000, 5]，目标域域数据输出300个rois[1, 300, 5]
        #print(rois)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        #print("输出这个值",self.training)
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))


            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)  #[1, 33300]

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))#通过的ne去除掉-1，返回非0的索引,即只要label不等于-1,则返回1，也就是此时返回的既有前景，也有背景
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)#从rpn_cls_score（b*12h*w，2)从第0轴按照rpn_keep索引找
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)#rpn_data上文就是tensor，不是Variable
            rpn_label = Variable(rpn_label.long())#运算完后的输出再用Variable( Tensor.long())转换回来

            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]


            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])
        #ROIs torch.Size([1, 2000, 5])训练似乎是多了一个索引值
        #ROIs torch.Size([1, 300, 5])测试

        return rois, self.rpn_loss_cls, self.rpn_loss_box

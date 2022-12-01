from __future__ import absolute_import

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------
'''
1.生成anchors，利用[dx(A),dy(A),dw(A),dh(A)]对所有的anchors做bbox regression回归（这里的anchors生成和训练时完全一致）

2.按照输入的foreground softmax scores由大到小排序anchors，提取前pre_nms_topN(e.g. 6000)个anchors，即提取修正位置后的foreground anchors。

3.限定超出图像边界的foreground anchors为图像边界（防止后续roi pooling时proposal超出图像边界）

4.剔除非常小（width<threshold or height<threshold）的foreground anchors

5.进行nonmaximum suppression

6.再次按照nms后的foreground softmax scores由大到小排序fg anchors，提取前post_nms_topN(e.g. 300)结果作为proposal输出。

7.输出的output的shape是[batch_size, 2000, 5]，在第3个维度上，第0列表示当前的region proposal属于batch size中的哪一张图像编号的，第1~4列表示region proposal在经过变换之后的输入图像分辨率上的坐标 [xmin,ymin,xmax,ymax]
'''

import torch
import torch.nn as nn
import numpy as np
import math
import yaml
from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
from model.nms.nms_wrapper import nms

import pdb

# 将RPN网络的每个anchor的分类得分以及检测框回归预估转换为目标候选
DEBUG = False


class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales),
                                                          ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    def forward(self, input):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs

        # 这里的input指的是(rpn_cls_prob.data, rpn_bbox_pred.data, im_info, cfg_key)   self._num_anchors=12
        # input[0]的维度1, 24, 37, 75
        scores = input[0][:, self._num_anchors:, :, :]  # 1, 12, 37, 75
        bbox_deltas = input[1]  ## input的第二维是偏移量[1, 48, 37, 75]
        im_info = input[2]  # input的第三维是图像的信息
        cfg_key = input[3]  # 用于指示是训练还是测试

        # 这里从config文件里提取了一些超参数
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N  # 源域是12000，目标域测试是6000 # 这个参数是在NMS处理之前我们要保留评分前多少的boxes
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N  # 训练是2000,测试是300 # 这个参数是在应用了NMS之后我们要保留前多少个评分boxes
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH  # 0.7 # 这个参数是NMS应用的阈值
        min_size = cfg[cfg_key].RPN_MIN_SIZE  # 16 # 这个参数是你最终映射回原图的宽和高都要大于这个值

        batch_size = bbox_deltas.size(0)  # 计算了一下偏移量的第一维，得到的是batch_size
        feat_height, feat_width = scores.size(2), scores.size(3)  # 在这里得到了rpn输出的H=37和W=75# 这两个数就是特征图的高度和宽度

        # 这里是要把他做成网格的形式，先做一个从0到75的数组，然后乘以stride，这样就是原图中
        # x的一系列坐标
        shift_x = np.arange(0, feat_width) * self._feat_stride  # self._feat_stride=16
        '''
        [0   16   32   48   64   80   96  112  128  144  160  176  192  208
         224  240  256  272  288  304  320  336  352  368  384  400  416  432
         448  464  480  496  512  528  544  560  576  592  608  624  640  656
         672  688  704  720  736  752  768  784  800  816  832  848  864  880
         896  912  928  944  960  976  992 1008 1024 1040 1056 1072 1088 1104
         1120 1136 1152 1168 1184]
        '''
        # 对y做同样的处理
        shift_y = np.arange(0, feat_height) * self._feat_stride
        '''
        [0  16  32  48  64  80  96 112 128 144 160 176 192 208 224 240 256 272
         288 304 320 336 352 368 384 400 416 432 448 464 480 496 512 528 544 560 576]
        '''
        # 这里是把x和y的坐标展开
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # 生成格点矩阵
        # 然后将xy进行合并，得到4*2775的结果，再进行转置，最终得到2775*4的维度
        # 然后将其转换为float的形式这些就是特征点转换到原图的中心点坐标
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel())).transpose())  # [2775*4]
        shifts = shifts.contiguous().type_as(scores).float()

        A = self._num_anchors  # 这里是12,faster rcnn为9
        K = shifts.size(0)  # K=height*width(特征图上的) # K是shifts的第一维，也就是2775，其实相当于在原图中划分了196个区域

        self._anchors = self._anchors.type_as(scores)
        # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()

        # 把以0, 0生成的那些标准anchor的坐标和原图中的中心点坐标相加，就得到了原图中的待选框
        # 这里计算出的结果是(2775, 12, 4)
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)  # shape[K,A,4] 得到所有的初始框[2775,12,4]
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)  # 把初始框的数组维度改变一下，变成[1,33300,4]
        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:

        # 这里的delta是(batch_size, 12*4, 37, 75)的大小，将他转换成和anchors一样的形式
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        # 转换为(batch_size, 37*75 * 4, 4)
        bbox_deltas = bbox_deltas.view(batch_size, -1,
                                       4)  # [1, 33300, 4] #将RPN输出的边框变换信息维度变回[N,H,W,C]，再改变一下维度，变成[1×H×W×A,4]

        # Same story for the scores:
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)  # 将RPN输出的分类信息维度变回[N,H,W,C]，再改变一下维度，变成[1×H×W×A,1]
        # Convert anchors into proposals via bbox transformations
        # 在这里结合RPN的输出变换初始框的坐标，得到第一次变换坐标后的proposals
        proposals = bbox_transform_inv(anchors, bbox_deltas,
                                       batch_size)  # bbox_transform_inv函数很简单，就是根据anchors和anchor的偏移量（tx, ty, tw, th）来生成 proposals

        # 22222222222222222222222222222222222222222222222222. clip predicted boxes to image
        # 在这里讲超出图像边界的proposal进行边界裁剪，使之在图像边界之内
        proposals = clip_boxes(proposals, im_info, batch_size)
        # proposals = clip_boxes_batch(proposals, im_info, batch_size)

        # assign the score to 0 if it's non keep.
        # keep = self._filter_boxes(proposals, min_size * im_info[:, 2])

        # trim keep index to make it euqal over batch
        # keep_idx = torch.cat(tuple(keep_idx), 0)

        # scores_keep = scores.view(-1)[keep_idx].view(batch_size, trim_size)
        # proposals_keep = proposals.view(-1, 4)[keep_idx, :].contiguous().view(batch_size, trim_size, 4)

        # _, order = torch.sort(scores_keep, 1, True)

        scores_keep = scores  # 这里先将scores存到keep里面(1, 37*75*12, 1)
        proposals_keep = proposals  # 这里是经过修正后的proposals[1, 33300, 4]
        _, order = torch.sort(scores_keep, 1,
                              True)  # 这里把是前景的分数进行排序，1代表以第2维进行排序，True代表从大到小# 返回的第一维是排好的tensor，第二维是index，这里只要index

        output = scores.new(batch_size, post_nms_topN,
                            5).zero_()  # 这里先定义了输出的tensor，一个(batch_size, post_nms_topN, 5)大小的全0矩阵
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]  # [33300, 4] 这里获取了一张图像的proposals
            scores_single = scores_keep[i]  # [33300,1]以及其是前景的评分

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 12000和)
            order_single = order[i]  # 这里计算出了一张图像的所有proposals前景评分从大到小的排名

            # 这里判断了一下，如果输入NMS的排序个数大于零，并且小于scores_keep的元素
            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            # 然后这里得到一个图像满足排名的proposal以及分数
            proposals_single = proposals_single[order_single, :]  # 对于源域数据选取前12000个proposal，目标域选取的是6000个
            scores_single = scores_single[order_single].view(-1, 1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            # 这里经过nms的操作得到这张图像保留下来的proposal
            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
            keep_idx_i = keep_idx_i.long().view(-1)

            # 这里取到的是经过nms保留下来的proposals以及他们的分数
            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            # 这里output的第三维之所以是5，因为第一维是加入了batch_size的序号，后面才是坐标#源域[1, 2000, 5]，目标域[1, 300, 5]
            num_proposal = proposals_single.size(0)
            output[i, :, 0] = i
            output[i, :num_proposal, 1:] = proposals_single
            # print("输出output的维度",output.shape)

        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1, 1).expand_as(ws)) & (hs >= min_size.view(-1, 1).expand_as(hs)))
        return keep

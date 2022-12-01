from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
#--------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------
'''
# 为每个anchor生成训练目标或标签，分类的标签只是0（非目标）1（是目标）-1（忽略）。当分类的标签大于0的时候预测框的回归才被指定。
'''
import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch

import pdb

DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3
'''
处理步骤为：
-->首先赋label：总的锚点为all_anchors，去除边框外的框
-->只算在边界内的框，这些框的下标inds_inside，框为anchors，分类为label=-1(不关心)，0（背景）,1（前景）。
-->将threshhold＞0.7的label=1，＜0.3的label=0。
-->还需要把与gt的overlap最大的框赋label=1。gt_argmax_overlaps是gt依次对应的第几个anchor。
-->各从前背景label 1 0中选取128个，其余重新设置label=-1
-->内部权重bbox_inside_weights 为坐标前乘的系数，格式为[[0,0,0,0或1111]，...刚开始为内部框个，后来是全部all_anchors个]
    外部权重bbox_outside_weights 为正负区别的系数，格式同上，不过非0数字为1/N_sample,正负权重可以设置（focal Loss论文中会设置）
-->前面都是在边界内anchor上计算的，
    然后用_unmap函数把label,bbox_targets,  bbox_inside_weights,bbox_outside_weights这几个参数扩展到
    全部anchor的shape尺度，对于边界外的anchor，label填充的是-1，其他三个填充的是0.（列数不变）
-->然后相当于有all_anchors个信息，这些信息的排序方式通过label.reshape((1, height, width, A))可以看出是
    联系特征图，先是A（K=12）个anchor为一组，按行排列，排列完一行另起一行。这样height, width, A就解释通了。
    数字1应该是1张图片
    然后.transpose(0,3,1,2)，可以把k理解为通道（只是假设理解），即先填充了一个通道的W×H吗，然后再填第二个通道
    最后.reshape((1, 1, A * height, width))顺序应该没变，只是括号[]前后位置变了
    其余3个返回值应该还是k-W-H-图的理解方式，只不过元素个数×4，也就是4-k-W-H-图
    .reshape((1, height, width, A * 4))
'''

'''
# 这个类主要对RPN的输出进行加工，对anchors打上标签，并且与ground_truth进行对比，计算他们之间的偏差
'''
class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        #这里的输入是(rpn_cls_score.data, gt_boxes, im_info, num_boxes)
        rpn_cls_score = input[0]#[1, 24, 37, 75]# 第一维是RPN分类得分,每个特征点生成12个anchor，要判断这12个究竟是属于前景还是背景，因此第二维为24
        gt_boxes = input[1]#[1, 50, 5]# 第二维是ground_truth (batch_size, gt的数量，5)5的前四维是坐标，最后一维是类别
        im_info = input[2] # 第三维是图像
        num_boxes = input[3]# 第四维是框的数量

        # map of shape (..., H, W)
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)

        batch_size = gt_boxes.size(0)

        # 又取了一遍，没啥用，还是获取特征框的高和宽
        # 和proposal_layer代码之前一样，还是将其还原到原图，然后做成网格的形式
        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride  #self._feat_stride=16
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()

        A = self._num_anchors  # A是anchors的数量，这里是12个
        K = shifts.size(0)# # K=height*width(特征图上的) # K是shifts的第一维，也就是2775


        self._anchors = self._anchors.type_as(gt_boxes) # move to specific gpu.
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        all_anchors = all_anchors.view(K * A, 4)  #[33300, 4]
        total_anchors = int(K * A)#[33300]
        # 这里判断了一下，过滤掉越界的边框，条件是左下角坐标必须大于0，右上角坐标小于图像的宽和高的最大值，这里允许边界框是压线的
        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))

        inds_inside = torch.nonzero(keep).view(-1)## 这里把所有不符合的anchors都过滤掉了，也就是越界的那些边框，得到符合规定的边框的索引,大约[17434]个
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]#[17434, 4]

        # 这里定义了三个label，1代表positive也就是前景，0代表背景，-1代表不关注
        # label: 1 is positive, 0 is negative, -1 is dont care
        # labels初始化，大小是(batch_size, 保留下来的框的数量)，初始用-1填补
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)#[1, 17434],全部填充-1
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()##[1, 17434],全部填充0 这个inside是论文中对正样本进行回归的参数，大小是(batch_size, 保留下的框的数量)，初始化为0
        bbox_outside_weights =gt_boxes.new(batch_size, inds_inside.size(0)).zero_()# #[1, 17434],全部填充0用来平衡RPN分类和回归的权重？？？？大小一样

        # 计算anchors和gt_boxes的IOU，返回的是(batch_size, 一个图中anchors的数量, 一个图中gt的数量)
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)#[1, 17434, 50]，个人理解：因为生成17434个anchors，同时又默认每个batch中的gtbox有50个，因此构成了一个17434x50 的矩阵，表示每个anchor和gtbox的IOU匹配值。

        # 这里获取了对于每一个batch，对应每个anchor最大的那个IOU，(batch_size, anchors的数量)
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)#[1, 17434]返回overlaps中的第三维数据中的最大值，第一个返回的tensor是具体的数值，第二个tensor是对应的索引值


        gt_max_overlaps, _ = torch.max(overlaps, 1)# [1, 50]这里返回对应每个gt，返回的第一个值为最大的IOU的值，第二个值为对应的索引值
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:# 这个参数默认是False 意思是先把符合负样本的标记为0
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # 如果gt_max_overlaps是0,则让他等于一个很小的值1*10^-5???
        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        #eq 相等返回1，不相等返回0
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)#[1, 17434]

        if torch.sum(keep) > 0:
            # 找出与gt相交最大且iou不为0的那个anchor，作为正样本
            labels[keep>0] = 1
        # fg label: above threshold IOU#cfg.TRAIN.RPN_POSITIVE_OVERLAP=0.7
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # 这里是前景需要的训练数量，前景占的比例 * 一个batch_size一共需要多少数量
        #系统设置的参数cfg.TRAIN.RPN_FG_FRACTION=0.5
        #系统设置的参数cfg.TRAIN.RPN_BATCHSIZE=256
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

        # 这里经过计算，得到目前已经确定的前景和背景的数量
        sum_fg = torch.sum((labels == 1).int(), 1)#大约为100-400左右
        sum_bg = torch.sum((labels == 0).int(), 1)#大约为16000-17000左右
        # 这里对一个batch_size进行迭代，看看选择的前景和背景数量是够符合规定要求
        for i in range(batch_size):
            # 如果得到的正样本太多，则需要二次采样
            # subsample positive labels if we have too many
            # 如果正样本的数量超过了预期的设置
            if sum_fg[i] > num_fg:
                # 首先获取所有的非零元素的索引
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                # 然后将他们用随机数的方式进行排列
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                # 这里就去前num_fg个作为正样本，其他的设置成-1也就是不关心
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

#           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]  #cfg.TRAIN.RPN_BATCHSIZE=256
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:

                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1

        offset = torch.arange(0, batch_size)*gt_boxes.size(1)#gt_boxes.size(1)为50,假设每个batch_size的gt_boxes都是50的话
        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)#[1, 17434]

        # 这里也相当于把gt_boxes给展开了
        # gt_boxes.view(-1, 5)相当于转换成(batch_size*50, 5)
        # argmax_overlaps.view(-1) ->(batch_size, anchor的数量)
        # gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :]这就是选出与每个anchorIOU最大的GT
        # 然后把anchors和与他们IOU最大的gt放入形参，计算他们之间的偏移量
        # 得到(batch_size, anchors的数量， 4)
        '''
        得到偏移量的真值后，将其保存在bbox_targets中。
        此同时， 还需要求解两个权值矩阵bbox_inside_weights和bbox_outside_weights，
        前者是用来设置正样本回归的权重，正样本设置为1，负样本设置为0，因为负样本对应的是背景， 不需要进行回归； 
        后者的作用则是平衡RPN分类损失与回归损失的权重，在此设置为1/256。
        '''
        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))#[1, 17434, 4]

        # use a single value instead of 4 values for easy index.
        # 所有前景的anchors，将他们的权重初始化
        bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]#[1, 17434]


        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))


        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights

        # 因为之前取labels的操作都是在对于图像范围内的边框进行的，这里要将图像外的都补成-1
        # 这样输出的就是和totalanchors一样大小的
        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)#[1, 33300]
        #print("大约有多少个正样本",torch.sum((labels == 1).int(), 1))
        # 同样，最其他的三个变量，也用相同的方式补全，这里是用0去填补
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)#[1, 33300, 4]
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)#[1, 33300]
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)#[1, 33300]

        outputs = []
        # 这里把labels变形了一下转换为(batch_size, 1, A * height, width)
        labels = labels.view(batch_size, height, width, A).permute(0,3,1,2).contiguous()

        labels = labels.view(batch_size, 1, A * height, width)
        #labels[1, 1, 444, 75]
        outputs.append(labels)

        bbox_targets = bbox_targets.view(batch_size, height, width, A*4).permute(0,3,1,2).contiguous()
        #bbox_targets[1, 48, 37, 75]
        outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        #bbox_inside_weights[1, 48, 37, 75]
        outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        #bbox_outside_weights[1, 48, 37, 75]
        outputs.append(bbox_outside_weights)
        #torch.sum((labels == 1).int(), 1)

        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])

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
#为每个目标候选生成训练目标或标签，分类标签从0 − K 0-K0−K（背景0或目标类别1 , … , K 1, \dots, K1,…,K），
# 自然lable值大于0的才被指定预测框回归。
import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
from ..utils.config import cfg
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch
import pdb

class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """
    """
       Assign object detection proposals to ground-truth targets. Produces proposal
       classification labels and bounding-box regression targets.
       这里将目标检测框分配给ground_truth，生成分类标签以及边框回归
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        # 这里还是进行初始化，对一些变量进行赋值
        self._num_classes = nclasses# 这是类别的数量
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)# 这里是进行标准化的均值和标准差
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)# 这里是进行标准化的均值和标准差
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS) # 这里有定义了一个权重，BBOX_INSIDE_WEIGHTS和上一个的RPN_BBOX_INSIDE_WEIGHTS有什么区别？

    def forward(self, all_rois, gt_boxes, num_boxes):
        # 重新定义了一下数据类型
        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)

        # 这里初始化了一个和gt_boxes一样大小的变量
        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        # 将gt_boxes的坐标一次赋给新的gt_boxes_append
        gt_boxes_append[:,:,1:5] = gt_boxes[:,:,:4]

        # Include ground-truth boxes in the set of candidate rois
        all_rois = torch.cat([all_rois, gt_boxes_append], 1)# 将rois和gt_boxes_append合并到一起#[1, 2050, 5]
        '''
                torch.cat  
                操作前   all_rois  shape [batch_size,2000,5]    gt_boxes_append   shape   [batch_size,50,5]
                操作后   all_rois  shape [batch_size,2050,5]    2050=num_region_proposal+num_max_gt      
        '''

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images) #256
        #print(cfg.TRAIN.BATCH_SIZE)
        '''
               # Minibatch size (number of regions of interest [ROIs])
               __C.TRAIN.BATCH_SIZE = 256  
               # Fraction of minibatch that is labeled foreground (i.e. class > 0)
               __C.TRAIN.FG_FRACTION = 0.25   
         '''
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))#64#选取0.25作为正样本
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        '''
                对于batch size中的每张训练图像，虽然会传给Fast R-CNN模型2000个region proposal
                但是每张图像中，Fast R-CNN模型只会训练256个正样本，其中包括小于等于64个正样本
                和大于等于256-64个负样本，再根据rois和gt_boxes对每张图像中所有的2000个region proposal
                进行正负样本的划分，对于batch size中的每张训练图像，从所有正样本region proposal中
                随机挑选出小于等于64个（如果region proposal中正样本的数量大于64，则随机挑选出64个，
                否则就把所有的正样本进行训练），然后在batch size中的每张图像从所有负样本中随机挑选出
                （256-对于当前图像所挑选出的正样本数）作为负样本(也就是192个负样本)，这里所指的正负样本是用于训练
                Fast R-CNN模型的region proposal，对于每张图像界定region proposal的正负样本的标准
                要依赖于当前训练图像的ground truth bounding boxes信息  

                在训练RPN阶段是需要在anchor boxes预选框的基础上进行位置调整，网络需要预测的也是相对于
                anchor boxes的坐标偏移量，根据当前图像gt_boxes信息对anchor boxes进行正负样本的划分
                计算RPN的分类损失和回归损失
                在训练Fast R-CNN阶段是需要在RPN输出的2000个region proposal基础上进行位置调整和预测坐标偏移量
                根据当前图像gt_boxes信息对region proposal进行正负样本的划分
                计算Fast R-CNN的分类损失和回归损失  
                '''

        labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois_pytorch(all_rois, gt_boxes, fg_rois_per_image,rois_per_image, self._num_classes)

        bbox_outside_weights = (bbox_inside_weights > 0).float()


        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS

        return bbox_targets, bbox_inside_weights


    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)

        targets = bbox_transform_batch(ex_rois, gt_rois)      #计算目标target的边框

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
                        / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets


    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)

        overlaps = bbox_overlaps_batch(all_rois, gt_boxes) #[1, 2050, 50]         #计算IOU
        """计算batch size中所有训练图像的region proposal与gt_boxes之间的overlap 
        overlaps  shape   [batch_size,2050,50]    2050=num_region_proposal+num_max_gt"""
        max_overlaps, gt_assignment = torch.max(overlaps, 2)

        '''对于batch size中的每张图像，RPN所给定的每个region proposal，遍历所有的gt_boxes
        得到当前region proposal与哪个gt_boxes的overlap最大，就认为当前的region proposal与
        gt_boxes的overlap是多少，region proposal的ground truth类别也与之对应'''


        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)

        offset = torch.arange(0, batch_size)*gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        labels = gt_boxes[:,:,4].contiguous().view(-1).index((offset.view(-1),)).view(batch_size, -1)

        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        rois_batch  = all_rois.new(batch_size, rois_per_image, 5).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):
            #print("这里的i等于o",i)
            #print("输出IOU",max_overlaps[i])

            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)## 这里计算了一下一张图像满足大于阈值的前景的数量
            #print("这里是前景的啥",fg_inds)
            fg_num_rois = fg_inds.numel()# 计算了一下有多少满足前景的rois


            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                                    (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
            bg_num_rois = bg_inds.numel()
            #print("背景目标大约", bg_num_rois)

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                '''
                    对于batch size中的每张图像，从所有正样本region proposal挑选出256*0.25=64
                    个正样本，（如果正样本的数量小于64）则将所有的正样本ROI都训练
                    SSD中正负样本的比例也是1：3
                    昨天看到一个帖子，训练Faster R-CNN的性能好不好关键并不在于对于RPN的训练，
                    而是在于对于Fast R-CNN的训练             
                '''

                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault. 
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_num_rois).long().cuda()
                '''进行了一个随其采样，得到随机采样的前景样本'''
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                #print("rand_num是什么",rand_num)
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
                #print("这里的fg_inds又是什么", fg_inds)
                fg_IOU=max_overlaps[i][fg_inds]
                #print("我猜这样就是输出IOU了",fg_IOU)
                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                # Seems torch.rand has a bug, it will generate very large number and make an error. 
                # We use numpy rand instead. 
                #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
                fg_IOU = max_overlaps[i][fg_inds]
                #print("我猜这样就是输出IOU了", fg_IOU)
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")
                
            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)

            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0

            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i,:,0] = i

            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]
        '''
                (1)rois_batch.shape,(batch_size, 256, 5), 
                用于对Fast R-CNN训练的rois，对于batch size中的每张训练图像随机选出了256个正负样本（比例1：3）region proposal
                其中5列的第一列表示当前的region proposal是属于batch size中哪张图像的图像索引编号
                后面4列表示所挑选出来的region proposal在输入图像空间分辨率上的位置坐标值
                这256个rois就是用于训练Fast R-CNN的，其中既有正样本也有负样本，但是在Fast R-CNN，是认为RPN所传送给它的
                2000个region proposal都是很好的（考虑过一定信息的）区域候选框
                (2)labels_batch.shape,(batch_size, 256),
                用于对Fast R-CNN训练的rois，对于batch size中的每张训练图像的256个region proposal的ground truth类别
                range   (0,num_classes-1)
                (3)gt_rois_batch.shape,(batch_size, 256, 5)
                用于对Fast R-CNN训练的rois，对于batch size中的每张训练图像的256个region proposal的坐标值ground truth（编码之前）
                注意前四列表示每个region proposal对应的ground truth boxes坐标值[xmin,ymin,xmax,ymax]还是在经过尺度变换的
                输入图像的空间分辨率上，最后一列表示rois对应的ground truth类别标签  0代表背景
                '''
        bbox_target_data = self._compute_targets_pytorch(
                rois_batch[:,:,1:5], gt_rois_batch[:,:,:4])

        bbox_targets, bbox_inside_weights = \
                self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)

        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights

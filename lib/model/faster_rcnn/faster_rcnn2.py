import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    def forward(self, d,im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data     #(size1,size2, image ratio(new image / source image) )
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        print("多少个目标",num_boxes)

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        print("这是变换前的", gt_boxes)
        x=torch.zeros(1,50,1).cuda()
        gt_boxes2=torch.cat((x,gt_boxes[:,:,0:4]),2)
        print("这是变换后的",gt_boxes2)
        print("变换后的特征维度",gt_boxes2.shape)


        # feed base feature map tp RPN to obtain rois
        # rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        # print("原始维度",rois)
        # print("roi的数据类型", type(rois))
        # # print("变换维度",rois.view(-1,5).shape)







        # if it is training phrase, then use ground trubut bboxes for refining
        # if self.training:
        #     roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
        #     rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
        #
        #     rois_label = Variable(rois_label.view(-1).long())
        #     rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
        #     rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
        #     rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        # else:
        # rois_label = None
        # rois_target = None
        # rois_inside_ws = None
        # rois_outside_ws = None
        # rpn_loss_cls = 0
        # rpn_loss_bbox = 0

        rois = Variable(rois)
        rois=torch.cat((x,gt_boxes[:,:,0:4]),2)
        # do roi pooling based on predicted rois


        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, gt_boxes2.view(-1, 5))

        print("输出pooled_feat的维度11111", pooled_feat.shape)
        print("输出pooled_feat的维度22222", pooled_feat[1,:,:,:].view(1,512,7,7).shape)
        # feed pooled features to top model
        for i in range(num_boxes):
            # print("总共多少个框",num_boxes)
            # print("这是第几个",i)
            # print("目标类别",gt_boxes[:,:,4][0,i])
            pooled_feat2 = self._head_to_tail((pooled_feat[i,:,:,:]).view(1,512,7,7))
            # print("输出pooled_feat的维度", pooled_feat2.shape)
            # if gt_boxes[:,:,4][0,i]==1:
            #     d.setdefault(1,[]).append(pooled_feat2)
            if gt_boxes[:,:,4][0,i]==6:
                d.setdefault(1,[]).append(pooled_feat2)

        # compute bbox offset
        # pooled_feat = self._head_to_tail(pooled_feat)
        # bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        # if self.training and not self.class_agnostic:
        #     # select the corresponding columns according to roi labels
        #     bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
        #     bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
        #     bbox_pred = bbox_pred_select.squeeze(1)
        #
        # # compute object classification probability
        # cls_score = self.RCNN_cls_score(pooled_feat)
        # cls_prob = F.softmax(cls_score, 1)
        #
        # RCNN_loss_cls = 0
        # RCNN_loss_bbox = 0
        #
        # if self.training:
        #     # classification loss
        #     RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
        #
        #     # bounding box regression L1 loss
        #     RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
        #
        #
        # cls_prob = cls_prob.view(batch_size, gt_boxes2.size(1), -1)
        # bbox_pred = bbox_pred.view(batch_size, gt_boxes2.size(1), -1)

        return gt_boxes2

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
    #
    def create_architecture(self):
        self._init_modules()
        self._init_weights()

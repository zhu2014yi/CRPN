# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# Reference: SiamRPN [Li]
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from  loss import  *

class CRPN(nn.Module):
    def __init__(self, anchors_nums=5, cls_type='thinner'):
        """
        :param cls_loss: thinner or thicker
                        thinner: output [B, 5, 17, 17] with BCE loss
                        thicker: output [B, 10, 17, 17] with nll loss
        """
        super(CRPN, self).__init__()
        self.features = None
        self.connect_model = None
        self.zf = None  # for online tracking
        self.anchor_nums = anchors_nums
        self.cls_type = cls_type
        self.criterion = nn.BCEWithLogitsLoss()
        self.anchors = None
        self.RPN = 2
        self.cfg=None

    def template(self, z):
        self.zf = self.feature_extractor(z)

    def feature_extractor(self, x):
        out, layer23_out = self.features(x)

        return out, layer23_out

    def track(self, x):
        zf, layer23_z_out = self.zf
        xf, layer23_x_out = self.feature_extractor(x)

        FTB1_z_out = self.FTB1_z(zf, layer23_z_out)
        FTB1_x_out = self.FTB1_x(xf, layer23_x_out)
        pred_clses,pred_reges=[],[]
        z, x = [zf, FTB1_z_out], [xf, FTB1_x_out]
        for i in range (self.RPN):
            pred_cls, pred_reg = self.connect_model[i](z[i], x[i])
            pred_clses.append(pred_cls)
            pred_reges.append(pred_reg)
        return pred_clses,pred_reges

    # ------- For Training ---------
    def _weight_l1_loss(self, pred_reg, label_reg, weight):
        """
        for reg loss (smooth l1 also works)
        """
        b, _, sh, sw = pred_reg.size()
        pred_reg = pred_reg.view(b, 4, -1, sh, sw)
        diff = (pred_reg - label_reg).abs()
        diff = diff.sum(dim=1).view(b, -1, sh, sw)
        loss = diff * weight
        return loss.sum().div(b)

    # cls loss thicker--
    def _loss_thicker(self, label_cls, label_reg, reg_weight, pred_cls, pred_reg,anchor_corner=None,bboxes_next=None,Giouloss=False):
        """
        cls loss and reg loss
        """
        b, c, h, w = pred_cls.size()
        pred_cls = pred_cls.view(b, 2, c // 2, h, w)
        pred_cls = pred_cls.permute(0, 2, 3, 4, 1).contiguous()
        pred_cls = F.log_softmax(pred_cls, dim=4)
        pred_cls = pred_cls.contiguous().view(-1, 2)

        cls_loss = self._weighted_CE(pred_cls, label_cls)
        if Giouloss:
            reg_loss = self.giou_loss(anchor_corner,bboxes_next,reg_weight)
        else:
            reg_loss = self._weight_l1_loss(pred_reg, label_reg, reg_weight)
        return cls_loss, reg_loss


    def _weighted_CE(self, pred, label):
        """
        for cls loss
        label_cls  -- 1: positive, 0: negative, -1: ignore
        """
        pred = pred.view(-1, 2)
        label = label.view(-1)
        pos = Variable(label.data.eq(1).nonzero().squeeze()).cuda()
        neg = Variable(label.data.eq(0).nonzero().squeeze()).cuda()

        loss_pos = self._cls_loss_thicker(pred, label, pos)
        loss_neg = self._cls_loss_thicker(pred, label, neg)
        return loss_pos * 0.5 + loss_neg * 0.5

    def _cls_loss_thicker(self, pred, label, select):
        if len(select.size()) == 0: return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)

        return F.nll_loss(pred, label)

    def forward(self, **kwargs):
        raise  NotImplementedError



    def giou_loss(self,pred, target, weights,eps=1e-6):
        """IoU loss.

        Computing the IoU loss between a set of predicted bboxes and target bboxes.
        The loss is calculated as negative log of IoU.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): Corresponding gt bboxes, shape (n, 4).
            eps (float): Eps to avoid log(0).

        Return:
            Tensor: Loss tensor.
        """
        gious = giou_bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
        #gious_test=gious.detach().cpu().numpy()
        loss = 1. - gious
        #loss_test = loss.reshape(16*17,5*17).detach().cpu().numpy()
        b, _, sh, sw = weights.size()
        loss=loss.view(b,-1,sh,sw).float()

        loss = loss * weights
        return loss.sum().div(b)





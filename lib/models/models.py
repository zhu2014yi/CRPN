# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# Details: This script provides main models proposed in CVPR2019 paper
#    1) SiamFCRes22: SiamFC with CIResNet-22 backbone
#    2) SiamFCRes22: SiamFC with CIResIncep-22 backbone
#    3) SiamFCNext22:SiamFC with CIResNext-22 backbone
#    4) SiamFCRes22W:Double 3*3 in the residual blob of CIResNet-22
# Main Results: see readme.md
# ------------------------------------------------------------------------------

from .siamfc import SiamFC_
from .siamrpn import SiamRPN_
from .C_RPN import CRPN
from .connect import Corr_Up, RPN_Up,DepthwiseRPN,AdjustLayer,FTB_v1,Adjustlayer_resnet
from .backbones import ResNet22, Incep22, ResNeXt22, ResNet22W
from  lib.dataset.utils import  *
eps = 1e-5
class SiamFCRes22(SiamFC_):
    def __init__(self, **kwargs):
        super(SiamFCRes22, self).__init__(**kwargs)
        self.features = ResNet22()
        self.connect_model = Corr_Up()


class SiamFCIncep22(SiamFC_):
    def __init__(self, **kwargs):
        super(SiamFCIncep22, self).__init__(**kwargs)
        self.features = Incep22()
        self.connect_model = Corr_Up()


class SiamFCNext22(SiamFC_):
    def __init__(self, **kwargs):
        super(SiamFCNext22, self).__init__(**kwargs)
        self.features = ResNeXt22()
        self.connect_model = Corr_Up()


class SiamFCRes22W(SiamFC_):
    def __init__(self, **kwargs):
        super(SiamFCRes22W, self).__init__(**kwargs)
        self.features = ResNet22W()
        self.connect_model = Corr_Up()

"""original """
# class SiamRPNRes22(SiamRPN_):
#     def __init__(self, **kwargs):
#         super(SiamRPNRes22, self).__init__(**kwargs)
#         self.features = ResNet22()
#         inchannels = self.features.feature_size
#         if self.cls_type == 'thinner': outchannels = 256
#         elif self.cls_type == 'thicker': outchannels = 512
#         else: raise ValueError('not implemented loss/cls type')
#         self.connect_model = RPN_Up(anchor_nums=self.anchor_nums,
#                                     inchannels=inchannels,
#                                     outchannels=outchannels,
#                                     cls_type = self.cls_type)

"""depthwise  0.387 """
class SiamRPNRes22(SiamRPN_):
    def __init__(self, **kwargs):
        super(SiamRPNRes22, self).__init__(**kwargs)
        self.features = ResNet22()
        inchannels = self.features.feature_size
        self.RPN = 2
        if self.cls_type == 'thinner': outchannels = 256
        elif self.cls_type == 'thicker': outchannels = 512
        else: raise ValueError('not implemented loss/cls type')
        self.connect_model = DepthwiseRPN()
        self.anchors=None
        self.cfg=None
    def track(self, x):
        xf ,_= self.feature_extractor(x)
        pred_cls, pred_reg = self.connector(self.zf[0], xf)
        return pred_cls, pred_reg
    def forward(self, template, search, bboxes=None, label_cls_total=None, label_reg=None, reg_weight=None,
                sum_weight=None):
        batch = template.shape[0]
        zf ,_= self.feature_extractor(template)
        xf,_ = self.feature_extractor(search)
        pred_cls, pred_reg = self.connector(zf, xf)
        #label_cls, label_cls_next = label_cls_total
        anchor_next = torch.tensor(self.anchors[1]).expand(batch, 4, 5, 17, 17).permute(0, 2, 3, 4,1).contiguous().view(-1,4)  # cx,cy,w,h
        anchor_next = torch.tensor(anchor_next, dtype=torch.float64).cuda()
        pred_reg_next = pred_reg.contiguous().view(batch, 4, 5, 17, 17).permute(0, 2, 3, 4, 1).contiguous().view(-1,4).double()
        bboxes_next = bboxes_expand(batch, bboxes)
        anchor_next = delta2boxes(anchor_next, pred_reg_next)
        anchor_corner=center2corner(anchor_next)
        label_cls_next = label_cls_total
        label_iou=label_cls_total.contiguous().view(-1)
        bboxes_next=bboxes_next[label_iou==1]
        anchor_corner=anchor_corner[label_iou==1]
        if self.training:
            if self.cls_type == 'thinner':
                cls_loss, reg_loss = self._loss(label_cls_next, label_reg, reg_weight, pred_cls, pred_reg, sum_weight)
            elif self.cls_type == 'thicker':
                cls_loss, reg_loss = self._loss_thicker(label_cls_next, label_reg, reg_weight, pred_cls, pred_reg,anchor_corner,bboxes_next,
                                                        Giouloss=self.cfg.SIAMRPN.TRAIN.GIoU_LOSS)
            else:
                raise ValueError('not implemented loss type')
            return cls_loss, reg_loss
        else:
            raise ValueError('forward is only used for training.')



"""CascadedSiamRPNRes22   EA0 best mean 0.395"""
class CascadedSiamRPNRes22(CRPN):
    def __init__(self, **kwargs):
        super(CascadedSiamRPNRes22, self).__init__(**kwargs)
        self.FTB1_z = FTB_v1(C=512, C_=512)
        self.FTB1_x = FTB_v1(C=512, C_=512)
        self.features = ResNet22()
        self.RPN = 2
        self.cfg=None
        self.connect_model=nn.ModuleList()
        for i in range(self.RPN):
            self.connect_model.append(DepthwiseRPN())
        self.anchors=None
    def forward(self,template, search, bboxes=None, label_cls=None, label_reg=None, reg_weight=None,
                sum_weight=None):
        batch = template.shape[0]

        zf, layer23_z_out = self.feature_extractor(template)
        xf, layer23_x_out = self.feature_extractor(search)
        FTB1_z_out = self.FTB1_z( zf,layer23_z_out)
        FTB1_x_out = self.FTB1_x( xf,layer23_x_out)

        """fix"""
        z, x = [zf, FTB1_z_out], [xf, FTB1_x_out]
        cls_losses, reg_losses = 0, 0
        for i in range(self.RPN):
            if i == 0:
                pred_cls, pred_reg = self.connect_model[i](z[i], x[i])
                cls_loss, reg_loss = self._loss_thicker(label_cls, label_reg, reg_weight, pred_cls, pred_reg)
                anchor_next = torch.tensor(self.anchors[1]).expand(batch, 4, 5, 17, 17).permute(0, 2, 3,4,1).contiguous().view(-1, 4)  # cx,cy,w,h
                anchor_next = torch.tensor(anchor_next, dtype=torch.float64).cuda()
                pred_reg_next = pred_reg.contiguous().view(batch, 4, 5, 17, 17).permute(0, 2, 3, 4,1).contiguous().view(-1,4).double()
                pred_cls_next = pred_cls.contiguous().view(batch, 2, 5, 17, 17).permute(0, 2, 3, 4,1).contiguous().view(-1,2).double()
                bboxes_next = bboxes_expand(batch, bboxes)
                anchor_next = delta2boxes(anchor_next, pred_reg_next)
                cls_losses += cls_loss
                reg_losses += reg_loss
            else:
                pred_cls, pred_reg = self.connect_model[i](z[i], x[i])
                """"select the way to filter sample"""
                label_cls_next, label_reg, reg_weight = Anchor_filter_and_relabel_fixpos(pred_cls_next,label_cls,bboxes_next,anchor_next)
                if self.Giouloss:
                    pred_reg_next = pred_reg.contiguous().view(batch, 4, 5, 17, 17).permute(0, 2, 3, 4,1).contiguous().view(-1,4).double()
                    pred_cls_next = pred_cls.contiguous().view(batch, 2, 5, 17, 17).permute(0, 2, 3, 4,1).contiguous().view(-1,2).double()
                    anchor_next = delta2boxes(anchor_next, pred_reg_next)
                    anchor_next_corner=center2corner(anchor_next)
                    cls_loss, reg_loss = self._loss_thicker(label_cls_next, label_reg, reg_weight, pred_cls,
                                                            pred_reg,anchor_next_corner,bboxes_next,Giouloss=self.cfg.SIAMRPN.TRAIN.GIoU_LOSS)
                else:
                    cls_loss, reg_loss = self._loss_thicker(label_cls_next, label_reg, reg_weight, pred_cls, pred_reg)
                cls_losses += cls_loss
                reg_losses += reg_loss
        return cls_losses, reg_losses


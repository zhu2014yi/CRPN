import torch
from torch.nn import functional as F
import  numpy as np
eps = 1e-5
import torch.nn as nn


def bboxes_expand(batch, bboxes):
    bboxes_next = []
    for i in range(batch):
        box = bboxes[i, :]
        boxes = box.expand(5 * 17 * 17, 4)
        bboxes_next.append(boxes)
    bboxes_next = torch.cat(bboxes_next, dim=0)
    return bboxes_next

def _dropout_esay_negative_threshold(neg,pos,drop,pred_cls, label_cls,b, threshold=0.9985):
    pred_cls = F.softmax(pred_cls, dim=1)
    pred_cls_neg = pred_cls[:, 0]
    mask_neg = torch.zeros_like(pred_cls_neg)
    mask_neg[neg[0]] = 1
    pred_cls_neg=pred_cls_neg*mask_neg
    pred_cls_neg[pred_cls_neg > threshold] = 0
    neg_idx=np.where(pred_cls_neg.detach().cpu().numpy()>0.0)
    label_cls=torch.ones_like(label_cls)*(-1)
    label_cls[pos[0]]=1
    label_cls[neg_idx[0]] = 0
    label_cls_batch=label_cls.contiguous().view(b,-1)
    return label_cls_batch


def _dropout_esay_negative_auto(neg,pos,drop,pred_cls,label_cls,b):
    """
    :param neg tuple
    :param pred_cls:(3125*b,2) torch.float32
    :param label_cls:(3125*b,) torch.int64
    :param B BATCH
    :param threshold:   None
    :return:
    """
    pred_cls = F.softmax(pred_cls, dim=1)
    pred_cls_neg=pred_cls[:,0]
    #pred_cls_neg_test=pred_cls_neg.detach().cpu().numpy()
    mask_neg=torch.zeros_like(pred_cls_neg)
    mask_pos=torch.zeros_like(pred_cls_neg)
    mask_drop=torch.zeros_like(pred_cls_neg)
    mask_neg[neg[0]]=1
    mask_pos[pos[0]]=1
    mask_drop[drop[0]]=1
    #total=mask_drop.sum()+mask_neg.sum()+mask_pos.sum()
    #mask1_test=mask_neg.detach().cpu().numpy()
    pred_cls_neg=pred_cls_neg*mask_neg
    #pred_cls_neg_test2=pred_cls_neg.detach().cpu().numpy()
    pred_cls_neg=pred_cls_neg+mask_pos+mask_drop
    #pred_cls_neg_test2=pred_cls_neg.detach().cpu().numpy()
    label_cls=torch.ones_like(label_cls)*(-1)

    label_cls[pos[0]]=1
    label_cls_batch=label_cls.contiguous().view(b,-1)
    pred_cls_neg_batch=pred_cls_neg.contiguous().view(b,-1)
    for i in range(b):
        neg_val,neg_idx=pred_cls_neg_batch[i].sort()
        neg_idx=neg_idx[:500]
        label_cls_batch[i][neg_idx]=0

    #label_cls=label_cls_batch.contiguous().view(-1)
    return label_cls_batch

def Anchor_filter_and_relabel( pred_cls, label_cls, bboxes, anchor_next, thr_up=0.6,
                                   thr_down=0.3):
    """

    :param pred_cls: N,4
    :parma label_cls: N,4
    :param bboxes: x1,y1,x2,y2: N,4
    :param Anchor_next:
    :param thr_up:
    :param thr_down:
    :return:
    """
    eps=1e-5
    b=label_cls.shape[0]

    center_boxes = corner2center(bboxes)
    delta = torch.zeros_like(anchor_next, dtype=torch.float32)
    # delta_weight=torch.zeros_like(delta)
    tcx, tcy, tw, th = center_boxes[:, 0], center_boxes[:, 1], center_boxes[:, 2], center_boxes[:, 3]
    center_pred_boxes = anchor_next
    cx, cy, w, h = center_pred_boxes[:, 0], center_pred_boxes[:, 1], center_pred_boxes[:, 2], center_pred_boxes[:,3]
    delta[:, 0] = (tcx - cx) / w
    delta[:, 1] = (tcy - cy) / h
    delta[:, 2] = torch.log(tw / (w + eps) + eps)
    delta[:, 3] = torch.log(th / (h + eps) + eps)
    # delta_test=delta.detach().numpy()wwwd
    anchor_corner = center2corner(anchor_next)
    overlap = bbox_overlaps(anchor_corner, bboxes, is_aligned=True)
    pos = np.where(overlap.detach().cpu() > thr_up)
    neg = np.where(overlap.detach().cpu() < thr_down)
    label_cls = torch.ones_like(overlap, dtype=torch.int64) * (-1)
    label_cls[neg[0]]=0
    label_cls[pos[0]]=1
    drop=np.where(label_cls.detach().cpu()==-1)
    """select the way to drop simple negative sample"""
    label_cls_batch = _dropout_esay_negative_threshold(neg, pos, drop, pred_cls, label_cls,b)  # input 1-D vector
    label_cls_next = torch.ones_like(label_cls_batch, dtype=torch.int64) * (-1)
    delta_weight = torch.zeros((b, label_cls_batch[0].shape[0]), dtype=torch.float32)

    for i in range(b):
        pos = label_cls_batch[i].data.eq(1).nonzero().squeeze()
        neg = label_cls_batch[i].data.eq(0).nonzero().squeeze()
        pos_c, pos_num = _select(pos.view(-1), 16)
        neg_c, neg_num = _select(neg.view(-1), 64 - pos_num)
        label_cls_next[i][pos_c] = 1
        label_cls_next[i][neg_c] = 0
        delta_weight[i][pos_c] = 1. / (pos_num + 1e-6)  # fix bugs here

    # pos_1=label_cls_next.view(-1).data.eq(1).nonzero().squeeze()
    label_cls_next = label_cls_next.contiguous().view(b, 5, 17, 17)
    delta_weight = delta_weight.contiguous().view(b, 5, 17, 17)
    delta = delta.contiguous().view(b, 5, 17, 17, 4).permute(0, 4, 1, 2, 3)
    return  label_cls_next.cuda(), delta.cuda(), delta_weight.cuda()

def Anchor_filter_and_relabel_fixpos( pred_cls, label_cls, bboxes, anchor_next, thr_up=0.6,
                                   thr_down=0.3):
    """

    :param pred_cls: N,4
    :parma label_cls: N,4
    :param bboxes: x1,y1,x2,y2: N,4
    :param Anchor_next:
    :param thr_up:
    :param thr_down:
    :return:
    """
    eps=1e-5
    b=label_cls.shape[0]
    pos=np.where(label_cls.contiguous().view(-1).detach().cpu()==1)
    center_boxes = corner2center(bboxes)
    delta = torch.zeros_like(anchor_next, dtype=torch.float32)
    # delta_weight=torch.zeros_like(delta)
    tcx, tcy, tw, th = center_boxes[:, 0], center_boxes[:, 1], center_boxes[:, 2], center_boxes[:, 3]
    center_pred_boxes = anchor_next
    cx, cy, w, h = center_pred_boxes[:, 0], center_pred_boxes[:, 1], center_pred_boxes[:, 2], center_pred_boxes[:,3]
    delta[:, 0] = (tcx - cx) / w
    delta[:, 1] = (tcy - cy) / h
    delta[:, 2] = torch.log(tw / (w + eps) + eps)
    delta[:, 3] = torch.log(th / (h + eps) + eps)
    # delta_test=delta.detach().numpy()wwwd
    anchor_corner = center2corner(anchor_next)
    overlap = bbox_overlaps(anchor_corner, bboxes, is_aligned=True)
    neg = np.where(overlap.detach().cpu() < thr_down)
    label_cls = torch.ones_like(overlap, dtype=torch.int64) * (-1)
    label_cls[neg[0]]=0
    label_cls[pos[0]]=1
    drop=np.where(label_cls.detach().cpu()==-1)
    neg=np.where(label_cls.detach().cpu()==0)
    """select the way to drop simple negative sample"""
    label_cls_batch = _dropout_esay_negative_threshold(neg, pos, drop, pred_cls, label_cls,b)  # input 1-D vector
    label_cls_next = torch.ones_like(label_cls_batch, dtype=torch.int64) * (-1)
    delta_weight = torch.zeros((b, label_cls_batch[0].shape[0]), dtype=torch.float32)

    for i in range(b):
        pos = label_cls_batch[i].data.eq(1).nonzero().squeeze()
        neg = label_cls_batch[i].data.eq(0).nonzero().squeeze()
        pos_c, pos_num = _select(pos.view(-1), 16)
        neg_c, neg_num = _select(neg.view(-1), 64 - pos_num)
        label_cls_next[i][pos_c] = 1
        label_cls_next[i][neg_c] = 0
        delta_weight[i][pos_c] = 1. / (pos_num + 1e-6)  # fix bugs here

    # pos_1=label_cls_next.view(-1).data.eq(1).nonzero().squeeze()
    label_cls_next = label_cls_next.contiguous().view(b, 5, 17, 17)
    delta_weight = delta_weight.contiguous().view(b, 5, 17, 17)
    delta = delta.contiguous().view(b, 5, 17, 17, 4).permute(0, 4, 1, 2, 3)
    return  label_cls_next.cuda(), delta.cuda(), delta_weight.cuda()


def _select(position, keep_num=16):
    """
    select pos and neg anchors to balance loss
    """
    num = position.shape[0]
    if num <= keep_num:
        return position, num
    slt = np.arange(num)
    np.random.shuffle(slt)
    slt = slt[:keep_num]
    return position[slt], keep_num

def center2corner(center):
    # shape: b,4,5,17,17
    x, y, w, h = center[:, 0], center[:, 1], center[:, 2], center[:, 3]
    x1 = x - w * 0.5
    x1 = x1[:, None, ...]
    y1 = y - h * 0.5
    y1 = y1[:, None, ...]
    x2 = x + w * 0.5
    x2 = x2[:, None, ...]
    y2 = y + h * 0.5
    y2 = y2[:, None, ...]
    if str(center.dtype) == 'torch.float32' or str(center.dtype) == 'torch.float64':
        return torch.cat((x1, y1, x2, y2), dim=1)
    else:
        return np.concatenate((x1, y1, x2, y2), axis=1)

def corner2center(corner):
    """
    [x1, y1, x2, y2] --> [cx, cy, w, h]
    """
    x1, y1, x2, y2 = corner[:, 0], corner[:, 1], corner[:, 2], corner[:, 3]
    x = (x1 + x2) * 0.5
    x = x[:, None, ...]
    y = (y1 + y2) * 0.5
    y = y[:, None, ...]
    w = x2 - x1
    w = w[:, None, ...]
    h = y2 - y1
    h = h[:, None, ...]
    if str(corner.dtype) == "torch.float64" or str(corner.dtype) == "torch.float32":
        return torch.cat((x, y, w, h), dim=1)
    else:
        return np.concatenate((x, y, w, h), axis=1)

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):

    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """
    """fix 小问题 w,h 减1 问题"""
    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt).clamp(min=0)  # [rows, 2]
        #wh_test=wh.detach().cpu().numpy()
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
                bboxes1[:, 3] - bboxes1[:, 1])

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
                    bboxes2[:, 3] - bboxes2[:, 1])
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
                bboxes1[:, 3] - bboxes1[:, 1])

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
                    bboxes2[:, 3] - bboxes2[:, 1])
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious
def giou_bbox_overlaps(bboxes1, bboxes2, is_aligned=False):
    '''Calculate generative overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.

    Returns:
        gious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    '''

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols
    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        overlaps_lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        overlaps_rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
        overlaps_wh = (overlaps_rb - overlaps_lt).clamp(min=0)  # [rows, 2]
        #overlaps_wh_test=overlaps_wh.detach().cpu().numpy()

        closure_lt = torch.min(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        closure_rb = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
        closure_wh = (closure_rb - closure_lt).clamp(min=0)  # [rows, 2]
        #closure_wh_test=closure_wh.detach().cpu().numpy()

        overlap = overlaps_wh[:, 0] * overlaps_wh[:, 1]
        closure = closure_wh[:, 0] * closure_wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] ) * (
            bboxes1[:, 3] - bboxes1[:, 1] )

        area2 = (bboxes2[:, 2] - bboxes2[:, 0] ) * (
            bboxes2[:, 3] - bboxes2[:, 1] )
        union = (area1 + area2 - overlap)
        ious = overlap / union
        gious = ious - (closure-union)/closure
    else:
       raise NotImplementedError
    return gious.clamp(min=0)
def delta2boxes(anchor_next, pred_reg_next):
    anchor_next_all = torch.zeros_like(anchor_next)
    anchor_next_all[:, 0] = anchor_next[:, 2] * pred_reg_next[:, 0] + anchor_next[:, 0]  # gx = px + pw * dx
    anchor_next_all[:, 1] = anchor_next[:, 3] * pred_reg_next[:, 1] + anchor_next[:, 1]  # gy = py + ph * dy
    anchor_next_all[:, 2] = anchor_next[:, 2] * torch.exp(pred_reg_next[:, 2])
    anchor_next_all[:, 3] = anchor_next[:, 3] * torch.exp(pred_reg_next[:, 3])
    return anchor_next_all

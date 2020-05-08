
import  torch
import  torch.nn as nn



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
        overlaps_wh = (overlaps_rb - overlaps_lt ).clamp(min=0)  # [rows, 2]

        closure_lt = torch.min(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        closure_rb = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        closure_wh = (closure_rb - closure_lt).clamp(min=0)  # [rows, 2]
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
        raise  NotImplementedError


    return gious

def giou_loss(pred, target, eps=1e-6):
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
    gious = giou_bbox_overlaps(pred, target, is_aligned=True)#.clamp(min=eps)
    loss = 1.-gious
    return loss


class GIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss





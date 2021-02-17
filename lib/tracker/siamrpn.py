# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# siamrpn class
# ------------------------------------------------------------------------------
import torch
import numpy as np

from torch.autograd import Variable
import torch.nn.functional as F
from utils.utils import load_yaml, get_subwindow_tracking, python2round, generate_anchor
from pysot.core.config import cfg
from pysot.tracker.classifier.base_classifier import BaseClassifier
from pysot.tracker.classifier.libs.plotting import show_tensor
from PIL import Image
from pysot.tracker.classifier.resnet import *
class SiamRPN(object):
    """
    modified from VOT18 released model
    """
    def __init__(self, info):
        super(SiamRPN, self).__init__()
        self.info = info   # model and benchmark info
        # if cfg.TRACK.USE_CLASSIFIER:
        #     self.classifier = BaseClassifier(self.model)

        #self.info.cls_type='thinner'
    def init(self, im, target_pos, target_sz, model, hp=None):
        self.frame_num = 1
        self.temp_max = 0
        self.lost_count = 0
        state = dict()
        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]
        p = RPNConfig()
        # single test
        if not hp and not self.info.epoch_test:
            prefix = [x for x in ['OTB', 'VOT'] if x in self.info.dataset]
            cfg_ = load_yaml('../experiments/test/{0}/{1}.yaml'.format(prefix[0], self.info.arch))
            cfg_benchmark = cfg_[self.info.dataset]
            p.update(cfg_benchmark)
            p.renew()

        # for vot17 or vot18: from siamrpn released
        if '2017' in self.info.dataset:
            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                p.instance_size = 287
                p.renew()
            else:
                p.instance_size = 271
                p.renew()

        # param tune
        if hp:
            p.update(hp)
            p.renew()

            # for small object (from DaSiamRPN released)
            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                p.instance_size = hp['big_sz']
                p.renew()
            else:
                p.instance_size = hp['small_sz']
                p.renew()


        net = model
        p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, p.score_size)

        avg_chans = np.mean(im, axis=(0, 1))

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = python2round(np.sqrt(wc_z * hc_z))

        z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

        z = Variable(z_crop.unsqueeze(0))
        net.template(z.cuda())

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]
        elif p.windowing == 'uniform':
            window = np.ones((p.score_size, p.score_size))
        window = np.expand_dims(window, axis=0)           # [1,17,17]
        window = np.repeat(window, p.anchor_num, axis=0)  # [5,17,17]
        if cfg.TRACK.USE_CLASSIFIER:
            # atom = ATOM().cuda().eval()
            # checkpoint_dict = torch.load("/home/zhuyi/Code/CRPN_511/atom_pretrain.pth")
            # atom.load_state_dict(checkpoint_dict["net"],strict=False)


            self.classifier = BaseClassifier(net)
            self.center_pos = np.array(target_pos)
            self.size = np.array(target_sz)
            bbox = [target_pos[0]-(target_sz[0]-1)/2,target_pos[1]-(target_sz[1]-1)/2,target_sz[0],target_sz[1]]
            #bbox = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
            if cfg.TRACK.TEMPLATE_UPDATE:
                with torch.no_grad():
                    net.template_short_term(self.z_crop)

            s_xx = s_z * (cfg.TRACK.INSTANCE_SIZE * 2 / cfg.TRACK.EXEMPLAR_SIZE)
            x_crop = get_subwindow_tracking(im, self.center_pos, cfg.TRACK.INSTANCE_SIZE * 2,
                round(s_xx), avg_chans)
            x_crop =x_crop.unsqueeze(0)
            self.classifier.initialize(x_crop.type(torch.FloatTensor), bbox)
        state['p'] = p
        state['net'] = net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz

        return state

    def update(self, net, x_crop, target_pos, target_sz, window, scale_z, p):
        score, delta = net.track(x_crop)
        # score=score[0]
        # delta=delta[0]
        b, c, s, s = delta.size()
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1, s, s).data.cpu().numpy()  # [4,5,17,17]
        if self.info.cls_type == 'thinner':
            score = torch.sigmoid(score).squeeze().cpu().data.numpy()  # [5,17,17]
        elif self.info.cls_type == 'thicker':
            score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1, s, s), dim=0).squeeze().data[1, ...].cpu().numpy()  # [5,17,17]

        delta[0, ...] = delta[0, ...] * p.anchor[2, ...] + p.anchor[0, ...]
        delta[1, ...] = delta[1, ...] * p.anchor[3, ...] + p.anchor[1, ...]
        delta[2, ...] = np.exp(delta[2, ...]) * p.anchor[2, ...]
        delta[3, ...] = np.exp(delta[3, ...]) * p.anchor[3, ...]

        # size penalty
        s_c = self.change(self.sz(delta[2, ...], delta[3, ...]) / (self.sz_wh(target_sz)))  # scale penalty  [5,17,17]
        r_c = self.change((target_sz[0] / target_sz[1]) / (delta[2, ...] / delta[3, ...]))  # ratio penalty  [5,17,17]

        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)  # [5,17,17]
        pscore = penalty * score  # [5, 17, 17]

        # window float
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence  # [5, 17, 17]
        a_max, r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        target = delta[:, a_max, r_max, c_max] / scale_z  # [4,1]

        target_sz = target_sz / scale_z
        lr = penalty[a_max, r_max, c_max] * score[a_max, r_max, c_max] * p.lr  # lr for OTB

        res_x = target[0] + target_pos[0]
        res_y = target[1] + target_pos[1]

        res_w = target_sz[0] * (1 - lr) + target[2] * lr
        res_h = target_sz[1] * (1 - lr) + target[3] * lr

        target_pos = np.array([res_x, res_y])
        target_sz = np.array([res_w, res_h])

        return target_pos, target_sz, score[a_max, r_max, c_max]
    def update_stage12(self, net, x_crop, target_pos, target_sz, window, scale_z, p, thr=0.9985):
        score, delta = net.track(x_crop)  # 1,10,17,17   4,5,17,17
        score = torch.cat(score, dim=0)  # 3,10,17,17
        delta = torch.cat(delta, dim=0)  # 3,20,17,17
        batch_cascaded, c, s, s = delta.size()  # 3,20,17,17
        delta = delta.contiguous().view(batch_cascaded, 4, 5, 17, 17). \
            permute(0, 2, 3, 4, 1).view(batch_cascaded, -1, 4).detach().cpu()  # [3,17*17*5,4]
        if self.info.cls_type == 'thinner':
            score = torch.sigmoid(score).squeeze().cpu().data.numpy()  # [5,17,17]
        elif self.info.cls_type == 'thicker':
            score = F.softmax(score.contiguous().view(batch_cascaded, 2, 5, 17, 17).permute(0, 2, 3, 4, 1). \
                              contiguous().view(batch_cascaded, -1, 2), dim=2).squeeze().detach().cpu()  # [3,5*17*17,2]

        anchors = torch.from_numpy(p.anchor).permute(1, 2, 3, 0).contiguous().view(-1, 4)

        def delta2boxes(anchor_next, pred_reg_next):
            anchor_next_all = torch.zeros_like(anchor_next)
            anchor_next_all[:, 0] = anchor_next[:, 2] * pred_reg_next[:, 0] + anchor_next[:, 0]  # gx = px + pw * dx
            anchor_next_all[:, 1] = anchor_next[:, 3] * pred_reg_next[:, 1] + anchor_next[:, 1]  # gy = py + ph * dy
            anchor_next_all[:, 2] = anchor_next[:, 2] * torch.exp(pred_reg_next[:, 2])
            anchor_next_all[:, 3] = anchor_next[:, 3] * torch.exp(pred_reg_next[:, 3])
            return anchor_next_all
        delta=torch.sum(delta,dim=0)

        pred_box = delta2boxes(anchors, delta)
        #pred_box = delta2boxes(anchors, delta[0])
        #pred_box = delta2boxes(pred_box, delta[1])

        pred_box = pred_box.contiguous().view(5, 17, 17, 4).permute(3, 0, 1, 2)  # 4,5,17,17
        # pred_box=pred_box.

        neg_idx = score[0, :, 0] > thr
        neg_idx = neg_idx.eq(1).nonzero().squeeze()
        score[1][neg_idx]=0

        # neg_idx = score[1, :, 0] > thr
        # neg_idx = neg_idx.eq(1).nonzero().squeeze()
        # score[2][neg_idx] = 0

        final_score=score[1]
        #final_score = score[2].contiguous().view(5, 17, 17, 2)
        final_score=final_score.contiguous().view(5,17,17,2)
        final_score = final_score[:, :, :, 1]  # 5 17 17
        #final_score_test=final_score.detach().cpu().numpy()
        # size penalty
        s_c = self.change(
            self.sz(pred_box[2, ...], pred_box[3, ...]) / (self.sz_wh(target_sz)))  # scale penalty  [5,17,17]
        r_c = self.change(
            (target_sz[0] / target_sz[1]) / (pred_box[2, ...] / pred_box[3, ...]))  # ratio penalty  [5,17,17]

        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)  # [5,17,17]
        # pscore = penalty * score  # [5, 17, 17]
        pscore = penalty * final_score  # [5, 17, 17]
        # window float
        pscore = pscore.detach().cpu().numpy()
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence  # [5, 17, 17]
        a_max, r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        target = pred_box[:, a_max, r_max, c_max] / scale_z  # [4,1]

        target_sz = target_sz / scale_z
        lr = penalty[a_max, r_max, c_max] * final_score[a_max, r_max, c_max] * p.lr  # lr for OTB

        res_x = target[0] + target_pos[0]
        res_y = target[1] + target_pos[1]

        res_w = target_sz[0] * (1 - lr) + target[2] * lr
        res_h = target_sz[1] * (1 - lr) + target[3] * lr

        target_pos = np.array([res_x, res_y])
        target_sz = np.array([res_w, res_h])

        return target_pos, target_sz, final_score[a_max, r_max, c_max]

    def update_stage12_mean(self, net, x_crop, target_pos, target_sz, window, scale_z, p, thr=0.9985):
        score, delta = net.track(x_crop)  # 1,10,17,17   4,5,17,17
        score = torch.cat(score, dim=0)  # 3,10,17,17
        delta = torch.cat(delta, dim=0)  # 3,20,17,17
        batch_cascaded, c, s, s = delta.size()  # 3,20,17,17
        delta = delta.contiguous().view(batch_cascaded, 4, 5, 17, 17). \
            permute(0, 2, 3, 4, 1).view(batch_cascaded, -1, 4).detach().cpu()  # [3,17*17*5,4]
        if self.info.cls_type == 'thinner':
            score = torch.sigmoid(score).squeeze().cpu().data.numpy()  # [5,17,17]
        elif self.info.cls_type == 'thicker':
            score = F.softmax(score.contiguous().view(batch_cascaded, 2, 5, 17, 17).permute(0, 2, 3, 4, 1). \
                              contiguous().view(batch_cascaded, -1, 2), dim=2).squeeze().detach().cpu()  # [3,5*17*17,2]

        anchors = torch.from_numpy(p.anchor).permute(1, 2, 3, 0).contiguous().view(-1, 4)

        def delta2boxes(anchor_next, pred_reg_next):
            anchor_next_all = torch.zeros_like(anchor_next)
            anchor_next_all[:, 0] = anchor_next[:, 2] * pred_reg_next[:, 0] + anchor_next[:, 0]  # gx = px + pw * dx
            anchor_next_all[:, 1] = anchor_next[:, 3] * pred_reg_next[:, 1] + anchor_next[:, 1]  # gy = py + ph * dy
            anchor_next_all[:, 2] = anchor_next[:, 2] * torch.exp(pred_reg_next[:, 2])
            anchor_next_all[:, 3] = anchor_next[:, 3] * torch.exp(pred_reg_next[:, 3])
            return anchor_next_all

        def normalize(score):
            score = (score - np.min(score)) / (np.max(score) - np.min(score))
            return score
        pred_box = delta2boxes(anchors, delta[0])
        pred_box = delta2boxes(pred_box, delta[1])

        pred_box = pred_box.contiguous().view(5, 17, 17, 4).permute(3, 0, 1, 2)  # 4,5,17,17
        # pred_box=pred_box.

        # neg_idx = score[0, :, 0] > thr
        # neg_idx = neg_idx.eq(1).nonzero().squeeze()
        # score[1][neg_idx]=0

        # neg_idx = score[1, :, 0] > thr
        # neg_idx = neg_idx.eq(1).nonzero().squeeze()
        # score[2][neg_idx] = 0

        final_score=(score[1]+score[0])/2
        #final_score=score[0]
        #final_score = score[2].contiguous().view(5, 17, 17, 2)
        final_score=final_score.contiguous().view(5,17,17,2)
        final_score = final_score[:, :, :, 1]  # 5 17 17
        #final_score_test=final_score.detach().cpu().numpy()
        # size penalty
        s_c = self.change(
            self.sz(pred_box[2, ...], pred_box[3, ...]) / (self.sz_wh(target_sz)))  # scale penalty  [5,17,17]
        r_c = self.change(
            (target_sz[0] / target_sz[1]) / (pred_box[2, ...] / pred_box[3, ...]))  # ratio penalty  [5,17,17]

        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)  # [5,17,17]
        # pscore = penalty * score  # [5, 17, 17]
        pscore = penalty * final_score  # [5, 17, 17]
        pscore = pscore.detach().cpu().numpy()

        if cfg.TRACK.USE_CLASSIFIER:
            #self.classifier.model.instance(x_crop)
            flag, s = self.classifier.track()
            if flag == 'not_found':
                self.lost_count += 1
            else:
                self.lost_count = 0

            confidence = Image.fromarray(s.detach().cpu().numpy())
            confidence = np.array(confidence.resize((17, 17)))
            pscore = pscore* (1 - cfg.TRACK.COEE_CLASS) + \
                     normalize(confidence) * cfg.TRACK.COEE_CLASS
            #pscore = pscore.flatten()

            if cfg.TRACK.TEMPLATE_UPDATE:
                score_st = self._convert_score(outputs['cls_st'])
                pred_bbox_st = self._convert_bbox(outputs['loc_st'], self.anchors)
                s_c_st = self.change(self.sz(pred_bbox_st[2, :], pred_bbox_st[3, :]) /
                                (self.sz(self.size[0] * scale_z, self.size[1] * scale_z)))
                r_c_st = self.change((self.size[0] / self.size[1]) /
                                (pred_bbox_st[2, :] / pred_bbox_st[3, :]))
                penalty_st = np.exp(-(r_c_st * s_c_st - 1) * cfg.TRACK.PENALTY_K)
                pscore_st = penalty_st * score_st
                pscore_st = pscore_st.reshape(5, -1) * (1 - cfg.TRACK.COEE_CLASS) + \
                            normalize(confidence) * cfg.TRACK.COEE_CLASS
                pscore_st = pscore_st.flatten(0)



        # window float

        pscore = pscore * (1 - p.window_influence) + window * p.window_influence  # [5, 17, 17]
        a_max, r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        target = pred_box[:, a_max, r_max, c_max] / scale_z  # [4,1]

        target_sz = target_sz / scale_z
        lr = penalty[a_max, r_max, c_max] * final_score[a_max, r_max, c_max] * p.lr  # lr for OTB

        if cfg.TRACK.USE_CLASSIFIER and cfg.TRACK.SHORT_TERM_DRIFT and self.lost_count >= 8:
            res_x, res_y = target[0] / 4 +target_pos[0], target[1] / 4 + target_pos[1]
        else:
            res_x = target[0] + target_pos[0]
            res_y = target[1] + target_pos[1]

        # res_x = target[0] + target_pos[0]
        # res_y = target[1] + target_pos[1]

        res_w = target_sz[0] * (1 - lr) + target[2] * lr
        res_h = target_sz[1] * (1 - lr) + target[3] * lr

        target_pos = np.array([res_x, res_y])
        target_sz = np.array([res_w, res_h])

        return target_pos, target_sz, final_score[a_max, r_max, c_max]

    def track(self, state, im):
        self.frame_num += 1
        self.curr_frame = im
        p = state['p']
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        wc_z = target_sz[1] + p.context_amount * sum(target_sz)
        hc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        # extract scaled crops for search region x at previous target position
        x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, python2round(s_x), avg_chans).unsqueeze(0))
        if state["arch"]=="SiamRPNRes22":
            target_pos, target_sz, score = self.update(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p)
        elif state["arch"]=="CascadedSiamRPNRes22":
            target_pos, target_sz, score = self.update_stage12_mean(net, x_crop.cuda(), target_pos, target_sz * scale_z, window,scale_z, p)
        else:
            raise  NotImplementedError
        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['score'] = score
        return state

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)


class RPNConfig(object):
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    windowing = 'cosine'
    exemplar_size = 127
    instance_size = 255
    total_stride = 8
    score_size = (instance_size - exemplar_size) // total_stride + 1
    context_amount = 0.5
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []

    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1
        self.anchor_num = len(self.ratios) * len(self.scales)




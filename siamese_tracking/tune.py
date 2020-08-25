from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import optuna
import logging
import cv2
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import  torch.nn.functional as F
from easydict import EasyDict as edict
from torch.autograd import Variable


from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from pysot.core.config import cfg
from toolkit.datasets import DatasetFactory, OTBDataset, UAVDataset, LaSOTDataset,VOTDataset, NFSDataset, VOTLTDataset
from toolkit.utils.region import vot_overlap, vot_float2str
from toolkit.evaluation import OPEBenchmark, EAOBenchmark, F1Benchmark

import lib.models.models as models
from lib.utils.utils import load_yaml, get_subwindow_tracking, python2round, generate_anchor,cxy_wh_2_rect

parser = argparse.ArgumentParser(description='tune siamrpn ')
parser.add_argument('--arch', dest='arch', default='CascadedSiamRPNRes22', help='backbone architecture')

parser.add_argument('--anchor_nums', default=5, type=int, help='anchor numbers')
parser.add_argument('--cls_type', default="thicker", type=str,
                    help='cls/loss type, thicker or thinner or else you defined')

parser.add_argument('--dataset', default='VOT2016', type=str, help='dataset')
parser.add_argument('--config', default='../experiments/test/VOT/CascadedSiamRPNRes22.yaml',
                    type=str, help='config file')
parser.add_argument('--snapshot', default='/2TB/zhuyi/Code/CRPN/snapshot_orig/checkpoint_e30.pth', type=str,
                    help='snapshot of models to eval')
parser.add_argument("--gpu_id", default="1", type=str, help="gpu id")

args = parser.parse_args()



class SiamRPN(object):
    """
    modified from VOT18 released model
    """
    def __init__(self, info):
        super(SiamRPN, self).__init__()
        self.info = info   # model and benchmark info
        #self.info.cls_type='thinner'
    def init(self, im, target_pos, target_sz, model, hp=None):
        state = dict()
        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]
        p = RPNConfig()

        # single test

        prefix = [x for x in ['OTB', 'VOT'] if x in self.info.dataset]
        cfg = load_yaml('../experiments/test/{0}/{1}.yaml'.format(prefix[0], self.info.arch))
        # cfg_benchmark = cfg[self.info.dataset]
        # p.update(cfg_benchmark)
        # p.renew()

        # for vot17 or vot18: from siamrpn released

        net = model
        p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, p.score_size)

        avg_chans = np.mean(im, axis=(0, 1))

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = python2round(np.sqrt(wc_z * hc_z))

        z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

        z = Variable(z_crop.unsqueeze(0))
        net.template(z.cuda())


        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]

        window = np.expand_dims(window, axis=0)           # [1,17,17]
        window = np.repeat(window, p.anchor_num, axis=0)  # [5,17,17]

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

        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)  # [5,17,17]
        pscore = penalty * score  # [5, 17, 17]

        # window float
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + window * cfg.TRACK.WINDOW_INFLUENCE  # [5, 17, 17]
        a_max, r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        target = delta[:, a_max, r_max, c_max] / scale_z  # [4,1]

        target_sz = target_sz / scale_z
        lr = penalty[a_max, r_max, c_max] * score[a_max, r_max, c_max] *cfg.TRACK.LR  # lr for OTB

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
        delta=torch.sum(delta,dim=0).div(2)

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

        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)  # [5,17,17]
        # pscore = penalty * score  # [5, 17, 17]
        pscore = penalty * final_score  # [5, 17, 17]
        # window float
        pscore = pscore.detach().cpu().numpy()
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + window * cfg.TRACK.WINDOW_INFLUENCE  # [5, 17, 17]
        a_max, r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        target = pred_box[:, a_max, r_max, c_max] / scale_z  # [4,1]

        target_sz = target_sz / scale_z
        lr = penalty[a_max, r_max, c_max] * final_score[a_max, r_max, c_max] * cfg.TRACK.LR  # lr for OTB

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

        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)  # [5,17,17]
        # pscore = penalty * score  # [5, 17, 17]
        pscore = penalty * final_score  # [5, 17, 17]
        # window float
        pscore = pscore.detach().cpu().numpy()
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + window * cfg.TRACK.WINDOW_INFLUENCE  # [5, 17, 17]
        a_max, r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        target = pred_box[:, a_max, r_max, c_max] / scale_z  # [4,1]

        target_sz = target_sz / scale_z
        lr = penalty[a_max, r_max, c_max] * final_score[a_max, r_max, c_max] * cfg.TRACK.LR # lr for OTB

        res_x = target[0] + target_pos[0]
        res_y = target[1] + target_pos[1]

        res_w = target_sz[0] * (1 - lr) + target[2] * lr
        res_h = target_sz[1] * (1 - lr) + target[3] * lr

        target_pos = np.array([res_x, res_y])
        target_sz = np.array([res_w, res_h])

        return target_pos, target_sz, final_score[a_max, r_max, c_max]

    def track(self, state, im):
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





def eval(dataset, tracker_name):
    # root = os.path.realpath(os.path.join(os.path.dirname(__file__),
    #                                      '../testing_dataset'))
    # root = os.path.join(root, dataset)
    tracker_dir = "./"
    trackers = [tracker_name]
    if 'OTB' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'LaSOT' == args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'UAV' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    elif 'NFS' in args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        eval_auc = benchmark.eval_success(tracker_name)
        auc = np.mean(list(eval_auc[tracker_name].values()))
        return auc
    if args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = EAOBenchmark(dataset)
        eval_eao = benchmark.eval(tracker_name)
        eao = eval_eao[tracker_name]['all']
        return eao
    elif 'VOT2018-LT' == args.dataset:
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = F1Benchmark(dataset)
        f1_result = {}
        with Pool(processes=num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                trackers), desc='eval f1', total=len(trackers), ncols=100):
                f1_result.update(ret)
        benchmark.show_result(f1_result,
                show_video_level=False)

    return 0


# fitness function
def objective(trial):
    # different params
    cfg.TRACK.WINDOW_INFLUENCE = trial.suggest_uniform('window_influence', 0.050, 0.650)
    cfg.TRACK.PENALTY_K = trial.suggest_uniform('penalty_k', 0.000, 0.600)
    cfg.TRACK.LR = trial.suggest_uniform('scale_lr', 0.100, 0.800)


    # rebuild tracker
    info = edict()
    info.arch = args.arch
    info.cls_type = args.cls_type
    info.dataset = args.dataset
    tracker = SiamRPN(info)


    model_name = args.snapshot.split('/')[-1].split('.')[0]
    tracker_name = os.path.join('tune_results', args.dataset, model_name, model_name + \
                                '_wi-{:.3f}'.format(cfg.TRACK.WINDOW_INFLUENCE) + \
                                '_pk-{:.3f}'.format(cfg.TRACK.PENALTY_K) + \
                                '_lr-{:.3f}'.format(cfg.TRACK.LR))
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                # if len(gt_bbox) == 4:
                #     gt_bbox = [gt_bbox[0], gt_bbox[1],
                #                gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                #                gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                #                gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    #gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    target_pos = np.array([cx, cy])
                    target_sz = np.array([w, h])
                    state=tracker.init(img, target_pos,target_sz,net)
                    state["arch"]=args.arch
                    # pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    state = tracker.track(state, img)  # track
                    location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

                    pred_bbox = location
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5  # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join(tracker_name, 'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                v_idx + 1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
        eao = eval(dataset=dataset_eval, tracker_name=tracker_name)
        info = "{:s} window_influence: {:1.17f}, penalty_k: {:1.17f}, scale_lr: {:1.17f}, EAO: {:1.3f}".format(
            model_name, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.PENALTY_K, cfg.TRACK.LR, eao)
        logging.getLogger().info(info)
        print(info)
        return eao
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                                          'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                                           '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path,
                                           '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                if not os.path.isdir(tracker_name):
                    os.makedirs(tracker_name)
                result_path = os.path.join(tracker_name, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx + 1, video.name, toc, idx / toc))
        auc = eval(dataset=dataset_eval, tracker_name=tracker_name)
        info = "{:s} window_influence: {:1.17f}, penalty_k: {:1.17f}, scale_lr: {:1.17f}, AUC: {:1.3f}".format(
            model_name, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.PENALTY_K, cfg.TRACK.LR, auc)
        logging.getLogger().info(info)
        print(info)
        return auc


if __name__ == "__main__":

    torch.set_num_threads(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # load config
    #cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    #dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)
    dataset_root = os.path.join("/ssd", args.dataset)
    # create model
    net = models.__dict__[args.arch](anchors_nums=args.anchor_nums, cls_type=args.cls_type)
    net = load_pretrain(net, args.snapshot)
    net.eval()
    net = net.cuda()

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    # Eval dataset
    # root = os.path.realpath(os.path.join(os.path.dirname(__file__),
    #                                      '../testing_dataset'))
    root="/ssd"
    root = os.path.join(root, args.dataset)
    if 'OTB' in args.dataset:
        dataset_eval = OTBDataset(args.dataset, root)
    elif 'LaSOT' == args.dataset:
        dataset_eval = LaSOTDataset(args.dataset, root)
    elif 'UAV' in args.dataset:
        dataset_eval = UAVDataset(args.dataset, root)
    elif 'NFS' in args.dataset:
        dataset_eval = NFSDataset(args.dataset, root)
    if args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        dataset_eval = VOTDataset(args.dataset, root)
    elif 'VOT2018-LT' == args.dataset:
        dataset_eval = VOTLTDataset(args.dataset, root)

    tune_result = os.path.join('tune_results', args.dataset)
    if not os.path.isdir(tune_result):
        os.makedirs(tune_result)
    log_path = os.path.join(tune_result, (args.snapshot).split('/')[-1].split('.')[0] + '.log')
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.FileHandler(log_path))
    optuna.logging.enable_propagation()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10000)
    print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))



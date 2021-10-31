from __future__ import print_function, absolute_import

import copy
import time
import random
import torch
from .utils.meters import AverageMeter
from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, TripletLossXBM, DivLoss, BridgeFeatLoss, BridgeProbLoss, AdvLoss
from torch import nn
import torch.nn.functional as F

from .models.layers.adain import adaptive_instance_normalization, adaptive_instance_normalization_v2


class Base_Trainer(object):
    def __init__(self, model, num_classes, margin=None):
        super(Base_Trainer, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, optimizer, print_freq=50, train_iters=400):
        # self.criterion_ce = CrossEntropyLabelSmooth(source_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        # losses_xbm = AverageMeter()
        precisions = AverageMeter()
        # precisions_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            # target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs = source_inputs['images'].cuda()
            targets = source_inputs['pids'].cuda()

            # t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            # device_num = torch.cuda.device_count()
            # B, C, H, W = s_inputs.size()
            #
            # def reshape(inputs):
            #     return inputs.view(device_num, -1, C, H, W)
            #
            # s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            # inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            # targets = targets.view(-1)
            # forward
            prob, feats = self._forward(inputs)
            # prob = prob[:, 0:source_classes + target_classes]

            # split feats
            # ori_feats = feats.view(device_num, -1, feats.size(-1))
            # feats_s, feats_t = ori_feats.split(ori_feats.size(1) // 2, dim=1)
            # ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            loss_ce = self.criterion_ce(prob, targets)
            loss_tri = self.criterion_tri(feats, targets)

            # enqueue and dequeue for xbm
            # if use_xbm:
            #     self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
            #     xbm_feats, xbm_targets = self.xbm.get()
            #     loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
            #     losses_xbm.update(loss_xbm.item())
            #     loss = loss_ce + loss_tri + loss_xbm
            # else:
            loss = loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ori_prob = prob.view(device_num, -1, prob.size(-1))
            # prob_s, prob_t = ori_prob.split(ori_prob.size(1) // 2, dim=1)
            # prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()
            prec, = accuracy(prob.view(-1, prob.size(-1)).data, targets.data)
            # prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions.update(prec[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_ce {:.3f} ({:.3f}) '
                      'Loss_tri {:.3f} ({:.3f}) '
                      'Prec_s {:.2%} ({:.2%}) '
                      .format(epoch, i + 1, len(data_loader_source),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg,
                              ))

    def _parse_data(self, inputs):
        imgs = inputs['images']
        pids = inputs['pids']
        indexes = inputs['indices']
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.model(inputs)


class DomainTraniner(object):
    def __init__(self, model, num_classes, margin=None):
        super(Base_Trainer, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, optimizer, print_freq=50, train_iters=400):
        # self.criterion_ce = CrossEntropyLabelSmooth(source_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        # losses_xbm = AverageMeter()
        precisions = AverageMeter()
        # precisions_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            # target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs = source_inputs['images'].cuda()
            targets = source_inputs['pids'].cuda()
            domains = source_inputs['domains'].cuda()

            # t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            # device_num = torch.cuda.device_count()
            # B, C, H, W = s_inputs.size()
            #
            # def reshape(inputs):
            #     return inputs.view(device_num, -1, C, H, W)
            #
            # s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            # inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            # targets = targets.view(-1)
            # forward
            prob, feats = self._forward(inputs)
            # prob = prob[:, 0:source_classes + target_classes]

            # split feats
            # ori_feats = feats.view(device_num, -1, feats.size(-1))
            # feats_s, feats_t = ori_feats.split(ori_feats.size(1) // 2, dim=1)
            # ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            loss_ce = self.criterion_ce(prob, targets)
            loss_tri = self.criterion_tri(feats, targets)

            # enqueue and dequeue for xbm
            # if use_xbm:
            #     self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
            #     xbm_feats, xbm_targets = self.xbm.get()
            #     loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
            #     losses_xbm.update(loss_xbm.item())
            #     loss = loss_ce + loss_tri + loss_xbm
            # else:
            loss = loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ori_prob = prob.view(device_num, -1, prob.size(-1))
            # prob_s, prob_t = ori_prob.split(ori_prob.size(1) // 2, dim=1)
            # prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()
            prec, = accuracy(prob.view(-1, prob.size(-1)).data, targets.data)
            # prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions.update(prec[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_ce {:.3f} ({:.3f}) '
                      'Loss_tri {:.3f} ({:.3f}) '
                      'Prec_s {:.2%} ({:.2%}) '
                      .format(epoch, i + 1, len(data_loader_source),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg,
                              ))

    def _parse_data(self, inputs):
        imgs = inputs['images']
        pids = inputs['pids']
        indexes = inputs['indices']
        domains = inputs['domains']
        return imgs.cuda(), pids.cuda(), indexes.cuda(), domains.cuda()

    def _forward(self, inputs):
        return self.model(inputs)


class Memory_Trainer(Base_Trainer):
    def __init__(self, model, num_classes, memory, margin=None):
        super(Memory_Trainer, self).__init__(model, num_classes, margin)
        self.memory = memory

    def train(self, epoch, data_loader_source, optimizer, print_freq=50, train_iters=400):
        # self.criterion_ce = CrossEntropyLabelSmooth(source_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_mem = AverageMeter()
        precisions = AverageMeter()
        # precisions_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, targets, _ = self._parse_data(source_inputs)

            # forward
            prob, feats, norm_feats = self._forward(inputs)

            # classification + triplet + memory
            loss_ce = self.criterion_ce(prob, targets)
            loss_tri = self.criterion_tri(feats, targets)
            loss_mem = self.memory(norm_feats, targets).mean()

            loss = loss_ce + loss_tri + loss_mem

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Updata memory classifier
            with torch.no_grad():
                imgs, pids, _ = self._parse_data(source_inputs)
                _, _, f_new = self.model(imgs)
                self.memory.module.MomentumUpdate(f_new, pids)

            prec, = accuracy(prob.view(-1, prob.size(-1)).data, targets.data)

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_mem.update(loss_mem.item())
            precisions.update(prec[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_ce {:.3f} ({:.3f}) '
                      'Loss_tri {:.3f} ({:.3f}) '
                      'Loss_mem {:.3f} ({:.3f}) '
                      'Prec_s {:.2%} ({:.2%}) '
                      .format(epoch, i + 1, len(data_loader_source),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              losses_mem.val, losses_mem.avg,
                              precisions.val, precisions.avg,
                              ))


class Baseline_Trainer(object):
    def __init__(self, model, xbm, num_classes, margin=None):
        super(Baseline_Trainer, self).__init__()
        self.model = model
        self.xbm = xbm
        self.num_classes = num_classes
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()
        self.criterion_tri_xbm = TripletLossXBM(margin=margin)

    def train(self, epoch, data_loader_source, data_loader_target, source_classes, target_classes,
              optimizer, print_freq=50, train_iters=400, use_xbm=False):
        self.criterion_ce = CrossEntropyLabelSmooth(source_classes + target_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_xbm = AverageMeter()
        precisions_s = AverageMeter()
        precisions_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            s_inputs, s_targets, _ = self._parse_data(source_inputs)
            t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            device_num = torch.cuda.device_count()
            B, C, H, W = s_inputs.size()

            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)

            s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            targets = targets.view(-1)
            # forward
            prob, feats = self._forward(inputs) 
            prob = prob[:, 0:source_classes + target_classes]
        
            # split feats
            ori_feats = feats.view(device_num, -1, feats.size(-1))
            feats_s, feats_t = ori_feats.split(ori_feats.size(1) // 2, dim=1)
            ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            loss_ce = self.criterion_ce(prob, targets)
            loss_tri = self.criterion_tri(ori_feats, targets)

            # enqueue and dequeue for xbm
            if use_xbm:
                self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
                xbm_feats, xbm_targets = self.xbm.get()
                loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
                losses_xbm.update(loss_xbm.item())
                loss = loss_ce + loss_tri + loss_xbm 
            else:
                loss = loss_ce + loss_tri 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ori_prob = prob.view(device_num, -1, prob.size(-1))
            prob_s, prob_t = ori_prob.split(ori_prob.size(1) // 2, dim=1)
            prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()
            prec_s, = accuracy(prob_s.view(-1, prob_s.size(-1)).data, s_targets.data)
            prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions_s.update(prec_s[0])
            precisions_t.update(prec_t[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                if use_xbm:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f}) '
                          'Data {:.3f} ({:.3f}) '
                          'Loss {:.3f} ({:.3f}) '
                          'Loss_ce {:.3f} ({:.3f}) '
                          'Loss_tri {:.3f} ({:.3f}) '
                          'Loss_xbm {:.3f} ({:.3f}) '
                          'Prec_s {:.2%} ({:.2%}) '
                          'Prec_t {:.2%} ({:.2%}) '
                          .format(epoch, i + 1, len(data_loader_target),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses.val, losses.avg,
                                  losses_ce.val, losses_ce.avg,
                                  losses_tri.val, losses_tri.avg,
                                  losses_xbm.val, losses_xbm.avg,
                                  precisions_s.val, precisions_s.avg,
                                  precisions_t.val, precisions_t.avg
                                  ))
                else:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f}) '
                          'Data {:.3f} ({:.3f}) '
                          'Loss {:.3f} ({:.3f}) '
                          'Loss_ce {:.3f} ({:.3f}) '
                          'Loss_tri {:.3f} ({:.3f}) '
                          'Prec_s {:.2%} ({:.2%}) '
                          'Prec_t {:.2%} ({:.2%}) '
                          .format(epoch, i + 1, len(data_loader_target),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses.val, losses.avg,
                                  losses_ce.val, losses_ce.avg,
                                  losses_tri.val, losses_tri.avg,
                                  precisions_s.val, precisions_s.avg,
                                  precisions_t.val, precisions_t.avg
                                  ))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.model(inputs)


class IDM_Trainer(object):
    def __init__(self, model, xbm, num_classes, margin=None, mu1=1.0, mu2=1.0, mu3=1.0):
        super(IDM_Trainer, self).__init__()
        self.model = model
        self.xbm = xbm
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.num_classes = num_classes
        self.criterion_ce = BridgeProbLoss(num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()
        self.criterion_tri_xbm = TripletLossXBM(margin=margin)
        self.criterion_bridge_feat = BridgeFeatLoss()
        self.criterion_diverse = DivLoss()

    def train(self, epoch, data_loader_source, data_loader_target, source_classes, target_classes,
              optimizer, print_freq=50, train_iters=400, use_xbm=False, stage=0):

        self.criterion_ce = BridgeProbLoss(source_classes + target_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_xbm = AverageMeter()
        losses_bridge_prob = AverageMeter()
        losses_bridge_feat = AverageMeter()
        losses_diverse = AverageMeter()
        
        precisions_s = AverageMeter()
        precisions_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            s_inputs, s_targets, _ = self._parse_data(source_inputs)
            t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            device_num = torch.cuda.device_count()
            B, C, H, W = s_inputs.size()

            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)

            s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            targets = targets.view(-1)
            # forward
            prob, feats, attention_lam= self._forward(inputs, stage) # attention_lam: [B, 2]
            prob = prob[:, 0:source_classes + target_classes]

            # split feats
            ori_feats = feats.view(device_num, -1, feats.size(-1))
            feats_s, feats_t, feats_mixed = ori_feats.split(ori_feats.size(1) // 3, dim=1)
            ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            loss_ce, loss_bridge_prob = self.criterion_ce(prob, targets, attention_lam[:,0].detach())
            loss_tri = self.criterion_tri(ori_feats, targets)
            loss_diverse = self.criterion_diverse(attention_lam)

            feats_s = feats_s.contiguous().view(-1, feats.size(-1))
            feats_t = feats_t.contiguous().view(-1, feats.size(-1))
            feats_mixed = feats_mixed.contiguous().view(-1, feats.size(-1))

            loss_bridge_feat = self.criterion_bridge_feat(feats_s, feats_t, feats_mixed, attention_lam)


            # enqueue and dequeue for xbm
            if use_xbm:
                self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
                xbm_feats, xbm_targets = self.xbm.get()
                loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
                losses_xbm.update(loss_xbm.item())
                loss = (1.-self.mu1) * loss_ce + loss_tri + loss_xbm + \
                       self.mu1 * loss_bridge_prob + self.mu2 * loss_bridge_feat + self.mu3 * loss_diverse
            else:
                loss = (1.-self.mu1) * loss_ce + loss_tri + \
                       self.mu1 * loss_bridge_prob + self.mu2 * loss_bridge_feat + self.mu3 * loss_diverse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ori_prob = prob.view(device_num, -1, prob.size(-1))
            prob_s, prob_t, _ = ori_prob.split(ori_prob.size(1) // 3, dim=1)

            prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()
            prec_s, = accuracy(prob_s.view(-1, prob_s.size(-1)).data, s_targets.data)
            prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_bridge_prob.update(loss_bridge_prob.item())
            losses_bridge_feat.update(loss_bridge_feat.item())
            losses_diverse.update(loss_diverse.item())
            
            precisions_s.update(prec_s[0])
            precisions_t.update(prec_t[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                if use_xbm:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f}) '
                          'Data {:.3f} ({:.3f}) '
                          'Loss {:.3f} ({:.3f}) '
                          'Loss_ce {:.3f} ({:.3f}) '
                          'Loss_tri {:.3f} ({:.3f}) '
                          'Loss_xbm {:.3f} ({:.3f}) '
                          'Loss_bridge_prob {:.3f} ({:.3f}) '
                          'Loss_bridge_feat {:.3f} ({:.3f}) '
                          'Loss_diverse {:.3f} ({:.3f}) '
                          'Prec_s {:.2%} ({:.2%}) '
                          'Prec_t {:.2%} ({:.2%}) '
                          .format(epoch, i + 1, len(data_loader_target),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses.val, losses.avg,
                                  losses_ce.val, losses_ce.avg,
                                  losses_tri.val, losses_tri.avg,
                                  losses_xbm.val, losses_xbm.avg,
                                  losses_bridge_prob.val, losses_bridge_prob.avg,
                                  losses_bridge_feat.val, losses_bridge_feat.avg,
                                  losses_diverse.val, losses_diverse.avg,
                                  precisions_s.val, precisions_s.avg,
                                  precisions_t.val, precisions_t.avg
                                  ))
                else:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f}) '
                          'Data {:.3f} ({:.3f}) '
                          'Loss {:.3f} ({:.3f}) '
                          'Loss_ce {:.3f} ({:.3f}) '
                          'Loss_tri {:.3f} ({:.3f}) '
                          'Loss_bridge_prob {:.3f} ({:.3f}) '
                          'Loss_bridge_feat {:.3f} ({:.3f}) '
                          'Loss_diverse {:.3f} ({:.3f}) '
                          'Prec_s {:.2%} ({:.2%}) '
                          'Prec_t {:.2%} ({:.2%}) '
                          .format(epoch, i + 1, len(data_loader_target),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses.val, losses.avg,
                                  losses_ce.val, losses_ce.avg,
                                  losses_tri.val, losses_tri.avg,
                                  losses_bridge_prob.val, losses_bridge_prob.avg,
                                  losses_bridge_feat.val, losses_bridge_feat.avg,
                                  losses_diverse.val, losses_diverse.avg,
                                  precisions_s.val, precisions_s.avg,
                                  precisions_t.val, precisions_t.avg
                                  ))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs, stage):
        return self.model(inputs, stage=stage)


class AdvTrainer(object):
    def __init__(self, model, num_classes, adv_update_steps=2, margin=None):
        super(AdvTrainer, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.adv_update_steps = adv_update_steps

        self.criterion_ce_lb = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_adv = AdvLoss()

    def train(self, epoch, data_loader_source, optimizers, print_freq=50, train_iters=400, phase=None):
        # self.criterion_ce = CrossEntropyLabelSmooth(source_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_dom_cls = AverageMeter()
        losses_disc = AverageMeter()
        losses_disc_dom = AverageMeter()
        losses_disc_con = AverageMeter()
        losses_adv = AverageMeter()
        losses_adv_dom = AverageMeter()
        losses_adv_con = AverageMeter()
        # losses_xbm = AverageMeter()
        precisions = AverageMeter()
        # precisions_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            # target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs = source_inputs['images'].cuda()
            targets = source_inputs['pids'].cuda()
            domains = source_inputs['domains'].cuda()

            # t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            # device_num = torch.cuda.device_count()
            # B, C, H, W = s_inputs.size()
            #
            # def reshape(inputs):
            #     return inputs.view(device_num, -1, C, H, W)
            #
            # s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            # inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            # targets = targets.view(-1)
            # forward

            prob, feats, domain_preds, domain_disc_preds, content_disc_preds = self._forward(inputs)

            if phase == 'adv' or (phase == 'mix' and i % 2 == 0):
                adv_domain_loss = F.cross_entropy(domain_disc_preds, domains)
                adv_content_loss = F.cross_entropy(content_disc_preds, targets)
                adv_loss = adv_domain_loss + adv_content_loss
                optimizers['adv'].zero_grad()
                adv_loss.backward()
                optimizers['adv'].step()
                losses_disc.update(adv_loss.item())
                losses_disc_con.update(adv_content_loss.item())
                losses_disc_dom.update(adv_domain_loss.item())

                if (i + 1) % print_freq == 0 or i % print_freq == 0:

                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f}) '
                          'Data {:.3f} ({:.3f}) '
                          'Disc Loss {:.3f} ({:.3f}) '
                          'Disc_dom Loss {:.3f} ({:.3f}) '
                          'Disc_con Loss {:.3f} ({:.3f}) '
                          .format(epoch, i + 1, len(data_loader_source),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses_disc.val, losses_disc.avg,
                                  losses_disc_dom.val, losses_disc_dom.avg,
                                  losses_disc_con.val, losses_disc_con.avg,
                              ))

            if phase == 'normal' or ((phase == 'mix' and i % 2 == 1)):
                # prob = prob[:, 0:source_classes + target_classes]

                # split feats
                # ori_feats = feats.view(device_num, -1, feats.size(-1))
                # feats_s, feats_t = ori_feats.split(ori_feats.size(1) // 2, dim=1)
                # ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

                # classification+triplet
                loss_ce = self.criterion_ce_lb(prob, targets)
                loss_tri = self.criterion_tri(feats, targets)
                loss_dom_cls = self.criterion_ce(domain_preds, domains)
                loss_adv_dom = self.criterion_adv(domain_disc_preds)
                loss_adv_con = self.criterion_adv(content_disc_preds)
                loss_adv = loss_adv_dom + loss_adv_con
                # loss_dom_cls = self.criterion_tri(domain_preds, domains)

                loss = loss_ce + loss_tri + loss_dom_cls + loss_adv
                # loss = loss_adv

                # loss = loss_ce + loss_tri + loss_dom_cls

                optimizers['comm'].zero_grad()
                loss.backward()
                optimizers['comm'].step()

                # ori_prob = prob.view(device_num, -1, prob.size(-1))
                # prob_s, prob_t = ori_prob.split(ori_prob.size(1) // 2, dim=1)
                # prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()
                prec, = accuracy(prob.view(-1, prob.size(-1)).data, targets.data)
                # prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

                losses.update(loss.item())
                losses_ce.update(loss_ce.item())
                losses_tri.update(loss_tri.item())
                losses_dom_cls.update(loss_dom_cls.item())
                losses_adv.update(loss_adv.item())
                losses_adv_dom.update(loss_adv_dom.item())
                losses_adv_con.update(loss_adv_con.item())

                precisions.update(prec[0])

                # print log
                batch_time.update(time.time() - end)
                end = time.time()

                if (i + 1) % print_freq == 0 or i % print_freq == 0:

                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f}) '
                          'Data {:.3f} ({:.3f}) '
                          'Loss {:.3f} ({:.3f}) '
                          'Loss_ce {:.3f} ({:.3f}) '
                          'Loss_tri {:.3f} ({:.3f}) '
                          'Loss_dom_cls {:.3f} ({:.3f}) '
                          'Loss_adv {:.3f} ({:.3f}) '
                          'Loss_adv_dom {:.3f} ({:.3f}) '
                          'Loss_adv_con {:.3f} ({:.3f}) '
                          'Prec_s {:.2%} ({:.2%}) '
                          .format(epoch, i + 1, len(data_loader_source),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg,
                                  losses.val, losses.avg,
                                  losses_ce.val, losses_ce.avg,
                                  losses_tri.val, losses_tri.avg,
                                  losses_dom_cls.val, losses_dom_cls.avg,
                                  losses_adv.val, losses_adv.avg,
                                  losses_adv_dom.val, losses_adv_dom.avg,
                                  losses_adv_con.val, losses_adv_con.avg,
                                  precisions.val, precisions.avg,
                              ))

    def _parse_data(self, inputs):
        imgs = inputs['images']
        pids = inputs['pids']
        indexes = inputs['indices']
        domains = inputs['domains']
        return imgs.cuda(), pids.cuda(), indexes.cuda(), domains.cuda()

    def _forward(self, inputs):
        return self.model(inputs)


class MoCoTrainer(object):
    def __init__(self, model, num_classes, margin=None):
        super(MoCoTrainer, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()
        self.criterion_moco = nn.CrossEntropyLoss().cuda()

    def train(self, epoch, data_loader_source, optimizer, print_freq=50, train_iters=400):
        # self.criterion_ce = CrossEntropyLabelSmooth(source_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_moco = AverageMeter()
        # losses_xbm = AverageMeter()
        precisions = AverageMeter()
        # precisions_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            # target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            # inputs = source_inputs['images'].cuda()
            targets = source_inputs['pids'].cuda()
            domains = source_inputs['domains'].cuda()

            # t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            # device_num = torch.cuda.device_count()
            # B, C, H, W = s_inputs.size()
            #
            # def reshape(inputs):
            #     return inputs.view(device_num, -1, C, H, W)
            #
            # s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            # inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            # targets = targets.view(-1)
            # forward
            prob, feats, moco_logits, moco_labels = self._forward(source_inputs)

            # prob = prob[:, 0:source_classes + target_classes]

            # split feats
            # ori_feats = feats.view(device_num, -1, feats.size(-1))
            # feats_s, feats_t = ori_feats.split(ori_feats.size(1) // 2, dim=1)
            # ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            loss_ce = self.criterion_ce(prob, targets)
            loss_tri = self.criterion_tri(feats, targets)
            loss_moco = self.criterion_moco(moco_logits, moco_labels)

            # enqueue and dequeue for xbm
            # if use_xbm:
            #     self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
            #     xbm_feats, xbm_targets = self.xbm.get()
            #     loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
            #     losses_xbm.update(loss_xbm.item())
            #     loss = loss_ce + loss_tri + loss_xbm
            # else:
            loss = loss_ce + loss_tri + loss_moco * 0.05

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ori_prob = prob.view(device_num, -1, prob.size(-1))
            # prob_s, prob_t = ori_prob.split(ori_prob.size(1) // 2, dim=1)
            # prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()
            prec, = accuracy(prob.view(-1, prob.size(-1)).data, targets.data)
            # prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_moco.update(loss_moco.item())
            precisions.update(prec[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_ce {:.3f} ({:.3f}) '
                      'Loss_tri {:.3f} ({:.3f}) '
                      'Loss_moco {:.3f} ({:.3f}) '
                      'Prec_s {:.2%} ({:.2%}) '
                      .format(epoch, i + 1, len(data_loader_source),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              losses_moco.val, losses_moco.avg,
                              precisions.val, precisions.avg,
                              ))

    def _parse_data(self, inputs):
        imgs = inputs['images']
        pids = inputs['pids']
        indexes = inputs['indices']
        domains = inputs['domains']
        return imgs.cuda(), pids.cuda(), indexes.cuda(), domains.cuda()

    def _forward(self, inputs):
        return self.model(inputs)


class RSCTrainer(object):
    def __init__(self, model, num_classes, margin=None):
        super(RSCTrainer, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, optimizer, print_freq=50, train_iters=400):
        # self.criterion_ce = CrossEntropyLabelSmooth(source_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        # losses_xbm = AverageMeter()
        precisions = AverageMeter()
        # precisions_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            # target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs = source_inputs['images'].cuda()
            targets = source_inputs['pids'].cuda()
            domains = source_inputs['domains'].cuda()

            # t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            # device_num = torch.cuda.device_count()
            # B, C, H, W = s_inputs.size()
            #
            # def reshape(inputs):
            #     return inputs.view(device_num, -1, C, H, W)
            #
            # s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            # inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            # targets = targets.view(-1)
            # forward
            prob, feats = self._forward(inputs, targets)
            # prob = prob[:, 0:source_classes + target_classes]

            # split feats
            # ori_feats = feats.view(device_num, -1, feats.size(-1))
            # feats_s, feats_t = ori_feats.split(ori_feats.size(1) // 2, dim=1)
            # ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            loss_ce = self.criterion_ce(prob, targets)
            loss_tri = self.criterion_tri(feats, targets)

            # enqueue and dequeue for xbm
            # if use_xbm:
            #     self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
            #     xbm_feats, xbm_targets = self.xbm.get()
            #     loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
            #     losses_xbm.update(loss_xbm.item())
            #     loss = loss_ce + loss_tri + loss_xbm
            # else:
            loss = loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ori_prob = prob.view(device_num, -1, prob.size(-1))
            # prob_s, prob_t = ori_prob.split(ori_prob.size(1) // 2, dim=1)
            # prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()
            prec, = accuracy(prob.view(-1, prob.size(-1)).data, targets.data)
            # prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions.update(prec[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_ce {:.3f} ({:.3f}) '
                      'Loss_tri {:.3f} ({:.3f}) '
                      'Prec_s {:.2%} ({:.2%}) '
                      .format(epoch, i + 1, len(data_loader_source),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg,
                              ))

    def _parse_data(self, inputs):
        imgs = inputs['images']
        pids = inputs['pids']
        indexes = inputs['indices']
        domains = inputs['domains']
        return imgs.cuda(), pids.cuda(), indexes.cuda(), domains.cuda()

    def _forward(self, inputs, targets):
        return self.model(inputs, targets)


class AttrTrainer(object):
    def __init__(self, model, num_classes, margin=None, lam=0.9):
        super(AttrTrainer, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()
        self.lam = lam

    def train(self, epoch, data_loader_source, optimizer, print_freq=50, train_iters=400):
        # self.criterion_ce = CrossEntropyLabelSmooth(source_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_attr = AverageMeter()
        # losses_xbm = AverageMeter()
        precisions = AverageMeter()
        # precisions_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            # target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs = source_inputs['images'].cuda()
            targets = source_inputs['pids'].cuda()
            attr_targets = source_inputs['attributes'].cuda()

            # t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            # device_num = torch.cuda.device_count()
            # B, C, H, W = s_inputs.size()
            #
            # def reshape(inputs):
            #     return inputs.view(device_num, -1, C, H, W)
            #
            # s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            # inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            # targets = targets.view(-1)
            # forward
            prob, feats, attr_preds = self._forward(inputs)

            loss_attr = 0

            num_attributes = len(attr_preds)
            for idx in range(num_attributes):
                loss_attr += F.cross_entropy(attr_preds[idx], attr_targets[:, idx])

            loss_attr /= num_attributes

            # prob = prob[:, 0:source_classes + target_classes]

            # split feats
            # ori_feats = feats.view(device_num, -1, feats.size(-1))
            # feats_s, feats_t = ori_feats.split(ori_feats.size(1) // 2, dim=1)
            # ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            loss_ce = self.criterion_ce(prob, targets)
            loss_tri = self.criterion_tri(feats, targets)

            # enqueue and dequeue for xbm
            # if use_xbm:
            #     self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
            #     xbm_feats, xbm_targets = self.xbm.get()
            #     loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
            #     losses_xbm.update(loss_xbm.item())
            #     loss = loss_ce + loss_tri + loss_xbm
            # else:

            loss = self.lam * (loss_ce + loss_tri) + (1 - self.lam) * loss_attr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ori_prob = prob.view(device_num, -1, prob.size(-1))
            # prob_s, prob_t = ori_prob.split(ori_prob.size(1) // 2, dim=1)
            # prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()
            prec, = accuracy(prob.view(-1, prob.size(-1)).data, targets.data)
            # prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_attr.update(loss_attr.item())
            precisions.update(prec[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_ce {:.3f} ({:.3f}) '
                      'Loss_tri {:.3f} ({:.3f}) '
                      'Loss_attr {:.3f} ({:.3f}) '
                      'Prec_s {:.2%} ({:.2%}) '
                      .format(epoch, i + 1, len(data_loader_source),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              losses_attr.val, losses_attr.avg,
                              precisions.val, precisions.avg,
                              ))

    def _parse_data(self, inputs):
        imgs = inputs['images']
        pids = inputs['pids']
        indexes = inputs['indices']
        domains = inputs['domains']
        return imgs.cuda(), pids.cuda(), indexes.cuda(), domains.cuda()

    def _forward(self, inputs):
        return self.model(inputs)


class SMMTrainer(object):
    def __init__(self, model, num_classes, margin=None):
        super(SMMTrainer, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, optimizer, print_freq=50, train_iters=400):
        # self.criterion_ce = CrossEntropyLabelSmooth(source_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_smm_ce = AverageMeter()
        losses_smm_tri = AverageMeter()
        # losses_xbm = AverageMeter()
        precisions = AverageMeter()
        # precisions_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            # target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs = source_inputs['images'].cuda()
            targets = source_inputs['pids'].cuda()
            domains = source_inputs['domains'].cuda()

            # swap the style of inputs
            batch_indices = torch.randperm(inputs.shape[0])
            style_inputs = inputs[batch_indices]
            lam = random.random()  # lam in range 0 to 1
            lam = lam * 0.5 + 0.5  # shrink to range 0.5 to 1
            mixed_inputs = adaptive_instance_normalization_v2(inputs, style_inputs, lam)

            # t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            # device_num = torch.cuda.device_count()
            # B, C, H, W = s_inputs.size()
            #
            # def reshape(inputs):
            #     return inputs.view(device_num, -1, C, H, W)
            #
            # s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            # inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            # targets = targets.view(-1)
            # forward
            prob, feats = self._forward(inputs)
            prob_smm, feats_smm = self._forward(mixed_inputs)
            # prob = prob[:, 0:source_classes + target_classes]

            # split feats
            # ori_feats = feats.view(device_num, -1, feats.size(-1))
            # feats_s, feats_t = ori_feats.split(ori_feats.size(1) // 2, dim=1)
            # ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            loss_ce = self.criterion_ce(prob, targets)
            loss_tri = self.criterion_tri(feats, targets)
            loss_smm_ce = self.criterion_ce(prob_smm, targets)
            loss_smm_tri = self.criterion_tri(feats_smm, targets)

            # enqueue and dequeue for xbm
            # if use_xbm:
            #     self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
            #     xbm_feats, xbm_targets = self.xbm.get()
            #     loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
            #     losses_xbm.update(loss_xbm.item())
            #     loss = loss_ce + loss_tri + loss_xbm
            # else:
            loss = loss_ce + loss_tri + (loss_smm_ce + loss_smm_tri) * 0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ori_prob = prob.view(device_num, -1, prob.size(-1))
            # prob_s, prob_t = ori_prob.split(ori_prob.size(1) // 2, dim=1)
            # prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()
            prec, = accuracy(prob.view(-1, prob.size(-1)).data, targets.data)
            # prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_smm_ce.update(loss_smm_ce.item())
            precisions.update(prec[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_ce {:.3f} ({:.3f}) '
                      'Loss_tri {:.3f} ({:.3f}) '
                      'Loss_smm_ce {:.3f} ({:.3f}) '
                      'Prec_s {:.2%} ({:.2%}) '
                      .format(epoch, i + 1, len(data_loader_source),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              losses_smm_ce.val, losses_smm_ce.avg,
                              precisions.val, precisions.avg,
                              ))

    def _parse_data(self, inputs):
        imgs = inputs['images']
        pids = inputs['pids']
        indexes = inputs['indices']
        domains = inputs['domains']
        return imgs.cuda(), pids.cuda(), indexes.cuda(), domains.cuda()

    def _forward(self, inputs):
        return self.model(inputs)


class SMMTrainer2(object):
    def __init__(self, model, num_classes, margin=None):
        super(SMMTrainer2, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, optimizer, print_freq=50, train_iters=400):
        # self.criterion_ce = CrossEntropyLabelSmooth(source_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        # losses_xbm = AverageMeter()
        precisions = AverageMeter()
        # precisions_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            # target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs = source_inputs['images'].cuda()
            targets = source_inputs['pids'].cuda()
            targets = torch.cat([targets, targets], dim=0)

            # t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            # device_num = torch.cuda.device_count()
            # B, C, H, W = s_inputs.size()
            #
            # def reshape(inputs):
            #     return inputs.view(device_num, -1, C, H, W)
            #
            # s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            # inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            # targets = targets.view(-1)
            # forward
            prob, feats = self._forward(inputs)
            prob = torch.cat(prob, dim=0)
            feats = torch.cat(feats, dim=0)

            # prob = prob[:, 0:source_classes + target_classes]

            # split feats
            # ori_feats = feats.view(device_num, -1, feats.size(-1))
            # feats_s, feats_t = ori_feats.split(ori_feats.size(1) // 2, dim=1)
            # ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            loss_ce = self.criterion_ce(prob, targets)
            loss_tri = self.criterion_tri(feats, targets)

            # enqueue and dequeue for xbm
            # if use_xbm:
            #     self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
            #     xbm_feats, xbm_targets = self.xbm.get()
            #     loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
            #     losses_xbm.update(loss_xbm.item())
            #     loss = loss_ce + loss_tri + loss_xbm
            # else:
            loss = loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ori_prob = prob.view(device_num, -1, prob.size(-1))
            # prob_s, prob_t = ori_prob.split(ori_prob.size(1) // 2, dim=1)
            # prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()
            prec, = accuracy(prob.view(-1, prob.size(-1)).data, targets.data)
            # prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions.update(prec[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_ce {:.3f} ({:.3f}) '
                      'Loss_tri {:.3f} ({:.3f}) '
                      'Prec_s {:.2%} ({:.2%}) '
                      .format(epoch, i + 1, len(data_loader_source),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg,
                              ))

    def _parse_data(self, inputs):
        imgs = inputs['images']
        pids = inputs['pids']
        indexes = inputs['indices']
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.model(inputs)


class SMMTrainerCY(object):
    def __init__(self, model, num_classes, margin=None):
        super(SMMTrainerCY, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, optimizer, print_freq=50, train_iters=400):
        # self.criterion_ce = CrossEntropyLabelSmooth(source_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_dom = AverageMeter()
        # losses_xbm = AverageMeter()
        precisions = AverageMeter()
        # precisions_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            # target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs = source_inputs['images'].cuda()
            targets = source_inputs['pids'].cuda()
            domains = source_inputs['domains'].cuda()
            targets = torch.cat([targets, targets], dim=0)

            # t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            # device_num = torch.cuda.device_count()
            # B, C, H, W = s_inputs.size()
            #
            # def reshape(inputs):
            #     return inputs.view(device_num, -1, C, H, W)
            #
            # s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            # inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            # targets = targets.view(-1)
            # forward

            prob, feats, domain_prob, domains = self._forward(inputs, domains)

            prob = torch.cat(prob, dim=0)
            feats = torch.cat(feats, dim=0)
            domain_prob = torch.cat(domain_prob, dim=0)
            domains = torch.cat(domains, dim=0)

            # prob = prob[:, 0:source_classes + target_classes]

            # split feats
            # ori_feats = feats.view(device_num, -1, feats.size(-1))
            # feats_s, feats_t = ori_feats.split(ori_feats.size(1) // 2, dim=1)
            # ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            loss_ce = self.criterion_ce(prob, targets)
            loss_tri = self.criterion_tri(feats, targets)
            loss_dom = F.cross_entropy(domain_prob, domains)

            # enqueue and dequeue for xbm
            # if use_xbm:
            #     self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
            #     xbm_feats, xbm_targets = self.xbm.get()
            #     loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
            #     losses_xbm.update(loss_xbm.item())
            #     loss = loss_ce + loss_tri + loss_xbm
            # else:
            loss = loss_ce + loss_tri + loss_dom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ori_prob = prob.view(device_num, -1, prob.size(-1))
            # prob_s, prob_t = ori_prob.split(ori_prob.size(1) // 2, dim=1)
            # prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()
            prec, = accuracy(prob.view(-1, prob.size(-1)).data, targets.data)
            # prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_dom.update(loss_dom.item())
            precisions.update(prec[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_ce {:.3f} ({:.3f}) '
                      'Loss_tri {:.3f} ({:.3f}) '
                      'Loss_dom {:.3f} ({:.3f}) '
                      'Prec_s {:.2%} ({:.2%}) '
                      .format(epoch, i + 1, len(data_loader_source),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              losses_dom.val, losses_dom.avg,
                              precisions.val, precisions.avg,
                              ))

    def _parse_data(self, inputs):
        imgs = inputs['images']
        pids = inputs['pids']
        indexes = inputs['indices']
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs, domains):
        return self.model(inputs, domains)


class MDETrainer(object):
    def __init__(self, model, num_classes, margin=None):
        super(MDETrainer, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.num_domains = len(self.num_classes)
        self.criterion_ce = [CrossEntropyLabelSmooth(num).cuda() for num in self.num_classes]
        self.criterion_tri = TripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, optimizer, print_freq=50, train_iters=400):
        # self.criterion_ce = CrossEntropyLabelSmooth(source_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        # losses_xbm = AverageMeter()
        precisions = AverageMeter()
        # precisions_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = [loader.next() for loader in data_loader_source]
            # target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs = [source['images'].cuda() for source in source_inputs]
            targets = [source['pids'].cuda() for source in source_inputs]

            # t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            # device_num = torch.cuda.device_count()
            # B, C, H, W = s_inputs.size()
            #
            # def reshape(inputs):
            #     return inputs.view(device_num, -1, C, H, W)
            #
            # s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            # inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            # targets = targets.view(-1)
            # forward
            prob, feats = self._forward(inputs)

            loss_ce = 0.
            loss_tri = 0.
            acc = 0.

            for domain_id, (cur_prob, cur_feat, target) in enumerate(zip(prob, feats, targets)):
                loss_ce += self.criterion_ce[domain_id](cur_prob, target)
                loss_tri += self.criterion_tri(cur_feat, target)
                acc += accuracy(cur_prob.view(-1, cur_prob.size(-1)).data, target.data)[0]

            loss_tri /= self.num_domains
            loss_ce /= self.num_domains
            acc /= self.num_domains

            # prob = prob[:, 0:source_classes + target_classes]

            # split feats
            # ori_feats = feats.view(device_num, -1, feats.size(-1))
            # feats_s, feats_t = ori_feats.split(ori_feats.size(1) // 2, dim=1)
            # ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            # loss_ce = self.criterion_ce(prob, targets)
            # loss_tri = self.criterion_tri(feats, targets)

            # enqueue and dequeue for xbm
            # if use_xbm:
            #     self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
            #     xbm_feats, xbm_targets = self.xbm.get()
            #     loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
            #     losses_xbm.update(loss_xbm.item())
            #     loss = loss_ce + loss_tri + loss_xbm
            # else:
            loss = loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ori_prob = prob.view(device_num, -1, prob.size(-1))
            # prob_s, prob_t = ori_prob.split(ori_prob.size(1) // 2, dim=1)
            # prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()

            # prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions.update(acc.cpu().numpy()[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_ce {:.3f} ({:.3f}) '
                      'Loss_tri {:.3f} ({:.3f}) '
                      'Prec_s {:.2%} ({:.2%}) '
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg,
                              ))

    def _parse_data(self, inputs):
        imgs = inputs['images']
        pids = inputs['pids']
        indexes = inputs['indices']
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.model(inputs)


class MLDGTrainer(object):
    def __init__(self, model, num_classes, num_domains=3, margin=None, mldg_beta=0.5):
        super(MLDGTrainer, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.criterion_ce = CrossEntropyLabelSmooth(self.num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()
        self.mldg_beta = mldg_beta

    def train(self, epoch, data_loader_source, optimizer, inner_opt_lr=0.00035, print_freq=50, train_iters=400):
        # self.criterion_ce = CrossEntropyLabelSmooth(source_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_inner = AverageMeter()
        losses_inner_ce = AverageMeter()
        losses_inner_tri = AverageMeter()
        losses_outer = AverageMeter()
        losses_outer_ce = AverageMeter()
        losses_outer_tri = AverageMeter()
        precisions = AverageMeter()
        precisions_inner = AverageMeter()
        precisions_outer = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = [loader.next() for loader in data_loader_source]

            test_domain = random.choice(range(self.num_domains))
            train_domains = [d for d in range(self.num_domains) if d != test_domain]

            data_time.update(time.time() - end)

            objective = 0.
            optimizer.zero_grad()
            for p in self.model.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

            '''
            inner update phase
            '''
            inner_net = copy.deepcopy(self.model)
            inner_net.train()
            # # inner_net_params = [p for p in inner_net.parameters() if p.requires_grad]
            # # outer_net_params = [p for p in self.model.parameters() if p.requires_grad]
            # inner_optimizer = torch.optim.Adam(inner_net.module.get_params(), lr=inner_opt_lr, weight_decay=5e-4)
            # train_inputs = [source_inputs[domain_id]['images'].cuda() for domain_id in train_domains]
            # train_targets = [source_inputs[domain_id]['pids'].cuda() for domain_id in train_domains]
            #
            # inner_prob, inner_feat = inner_net(train_inputs)
            # inner_prob = torch.cat(inner_prob, dim=0)
            # inner_feat = torch.cat(inner_feat, dim=0)
            # inner_targets = torch.cat(train_targets, dim=0)
            #
            # loss_inner_ce = self.criterion_ce(inner_prob, inner_targets)
            # loss_inner_tri = self.criterion_tri(inner_feat, inner_targets)
            # acc_inner = accuracy(inner_prob.view(-1, inner_prob.size(-1)).data, inner_targets.data)[0]
            # loss_inner = loss_inner_ce + loss_inner_tri
            #
            # inner_optimizer.zero_grad()
            # loss_inner.backward()
            # inner_optimizer.step()
            #
            # # Now inner_net has accumulated gradients Gi
            # # The clone-network (inner_net) has now parameters P - lr * Gi
            # # Adding the gradient Gi to original network
            # for p_outer, p_inner in zip(self.model.module.get_params(), inner_net.module.get_params()):
            #     if p_inner.grad is not None:
            #         assert p_inner.grad.data.shape == p_outer.grad.data.shape
            #         p_outer.grad.data.add_(p_inner.grad.data / self.num_domains)
            #
            # objective += loss_inner.item()

            '''
            outer update phase
            '''
            test_inputs = [source_inputs[test_domain]['images'].cuda()]
            test_targets = [source_inputs[test_domain]['pids'].cuda()]

            outer_prob, outer_feat = inner_net(test_inputs)
            # outer_prob, outer_feat = self.model(test_inputs)

            outer_prob = torch.cat(outer_prob, dim=0)
            outer_feat = torch.cat(outer_feat, dim=0)
            test_targets = torch.cat(test_targets, dim=0)

            loss_outer_ce = self.criterion_ce(outer_prob, test_targets)
            loss_outer_tri = self.criterion_tri(outer_feat, test_targets)
            acc_outer = accuracy(outer_prob.view(-1, outer_prob.size(-1)).data, test_targets.data)[0]
            loss_outer = loss_outer_ce + loss_outer_tri

            loss_outer.backward()

            for p_outer, p_inner in zip(self.model.parameters(), inner_net.parameters()):
                if p_inner.grad is not None:
                    assert p_inner.grad.data.shape == p_outer.grad.data.shape
                    p_outer.grad.data.add_(p_inner.grad.data / self.num_domains)

            for (n_outer, m_outer), (n_inner, m_inner) in zip(self.model.named_modules(), inner_net.named_modules()):
                assert n_outer == n_inner
                if isinstance(m_outer, nn.BatchNorm2d):
                    m_outer.running_mean.data = m_inner.running_mean.data.clone()
                    m_outer.running_var.data = m_inner.running_var.data.clone()
                if isinstance(m_outer, nn.BatchNorm1d):
                    m_outer.running_mean.data = m_inner.running_mean.data.clone()
                    m_outer.running_var.data = m_inner.running_var.data.clone()

            # for p_outer, p_inner in zip(self.model.named_parameters(), inner_net.named_parameters()):
            #     assert p_outer[0] == p_inner[0] and p_outer[1].shape == p_inner[1].shape
            #     if isinstance(p_outer[1], nn.BatchNorm1d):
            #         p_outer[1].running_mean.data = p_inner[1].running_mean.data.clone()
            #         p_outer[1].running_var.data = p_inner[1].running_var.data.clone()
            #     if isinstance(p_outer[1], nn.BatchNorm2d):
            #         p_outer[1].running_mean.data = p_inner[1].running_mean.data.clone()
            #         p_outer[1].running_var.data = p_inner[1].running_var.data.clone()

            # for p_outer, p_inner in zip(self.model.module.get_params(), inner_net.module.get_params()):
            #     if p_inner.grad is not None:
            #         assert p_inner.grad.data.shape == p_outer.grad.data.shape
            #         p_outer.grad.data.add_(p_inner.grad.data / self.num_domains)

            # grad_outer = torch.autograd.grad(loss_outer, inner_net.module.get_params(), allow_unused=True)
            #
            # objective += self.mldg_beta * loss_outer.item()
            #
            # for p_outer, g_outer in zip(self.model.module.get_params(), grad_outer):
            #     if g_outer is not None:
            #         assert g_outer.data.shape == p_outer.grad.data.shape
            #         p_outer.grad.data.add_(self.mldg_beta * g_outer.data / self.num_domains)

            optimizer.step()

            # with torch.no_grad():
            #     self.model.eval()
            #     prob, feats = self.model(test_inputs)
            #     prob = torch.cat(prob, dim=0)
            #     acc_test = accuracy(prob.view(-1, outer_prob.size(-1)).data, test_targets.data)[0]
            #     self.model.train()


            # process inputs
            # inputs = [source['images'].cuda() for source in source_inputs]
            # targets = [source['pids'].cuda() for source in source_inputs]

            # t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            # device_num = torch.cuda.device_count()
            # B, C, H, W = s_inputs.size()
            #
            # def reshape(inputs):
            #     return inputs.view(device_num, -1, C, H, W)
            #
            # s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            # inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            # targets = targets.view(-1)
            # forward
            # prob, feats = self._forward(inputs)

            # loss_ce = 0.
            # loss_tri = 0.
            # acc = 0.

            # for domain_id, (cur_prob, cur_feat, target) in enumerate(zip(prob, feats, targets)):
            #     loss_ce += self.criterion_ce[domain_id](cur_prob, target)
            #     loss_tri += self.criterion_tri(cur_feat, target)
            #     acc += accuracy(cur_prob.view(-1, cur_prob.size(-1)).data, target.data)[0]

            # acc /= len(self.num_classes)

            # prob = prob[:, 0:source_classes + target_classes]

            # split feats
            # ori_feats = feats.view(device_num, -1, feats.size(-1))
            # feats_s, feats_t = ori_feats.split(ori_feats.size(1) // 2, dim=1)
            # ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            # loss_ce = self.criterion_ce(prob, targets)
            # loss_tri = self.criterion_tri(feats, targets)

            # enqueue and dequeue for xbm
            # if use_xbm:
            #     self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
            #     xbm_feats, xbm_targets = self.xbm.get()
            #     loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
            #     losses_xbm.update(loss_xbm.item())
            #     loss = loss_ce + loss_tri + loss_xbm
            # else:
            # loss = loss_ce + loss_tri

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # ori_prob = prob.view(device_num, -1, prob.size(-1))
            # prob_s, prob_t = ori_prob.split(ori_prob.size(1) // 2, dim=1)
            # prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()

            # prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(objective)
            # losses_inner_ce.update(loss_inner_ce.item())
            # losses_inner_tri.update(loss_inner_tri.item())
            # losses_inner.update(loss_inner.item())
            losses_outer_ce.update(loss_outer_ce.item())
            losses_outer_tri.update(loss_outer_tri.item())
            losses_outer.update(loss_outer.item())
            # precisions_inner.update(acc_inner.cpu().numpy()[0])
            precisions_outer.update(acc_outer.cpu().numpy()[0])
            # precisions.update((acc_inner.cpu().numpy()[0] + acc_outer.cpu().numpy()[0]) / 2)
            # precisions.update(acc_test.cpu().numpy()[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_inner {:.3f} ({:.3f}) '
                      'Loss_inner_ce {:.3f} ({:.3f}) '
                      'Loss_inner_tri {:.3f} ({:.3f}) '
                      'Loss_outer {:.3f} ({:.3f}) '
                      'Loss_outer_ce {:.3f} ({:.3f}) '
                      'Loss_outer_tri {:.3f} ({:.3f}) '
                      'Acc {:.2%} ({:.2%}) '
                      'Acc_inner {:.2%} ({:.2%}) '
                      'Acc_outer {:.2%} ({:.2%}) '
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_inner.val, losses_inner.avg,
                              losses_inner_ce.val, losses_inner_ce.avg,
                              losses_inner_tri.val, losses_inner_tri.avg,
                              losses_outer.val, losses_outer.avg,
                              losses_outer_ce.val, losses_outer_ce.avg,
                              losses_outer_tri.val, losses_outer_tri.avg,
                              precisions.val, precisions.avg,
                              precisions_inner.val, precisions_inner.avg,
                              precisions_outer.val, precisions_outer.avg,
                              ))

    def _parse_data(self, inputs):
        imgs = inputs['images']
        pids = inputs['pids']
        indexes = inputs['indices']
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.model(inputs)


class MLDGTrainer2(object):
    def __init__(self, model, num_classes, num_domains=3, margin=None, mldg_beta=0.5):
        super(MLDGTrainer2, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.criterion_ce = CrossEntropyLabelSmooth(self.num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()
        self.mldg_beta = mldg_beta

    def train(self, epoch, data_loader_source, optimizer, print_freq=50, train_iters=400):
        # self.criterion_ce = CrossEntropyLabelSmooth(source_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_inner = AverageMeter()
        losses_inner_ce = AverageMeter()
        losses_inner_tri = AverageMeter()
        losses_outer = AverageMeter()
        losses_outer_ce = AverageMeter()
        losses_outer_tri = AverageMeter()
        precisions = AverageMeter()
        precisions_inner = AverageMeter()
        precisions_outer = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = [loader.next() for loader in data_loader_source]

            test_domain = random.choice(range(self.num_domains))
            train_domains = [d for d in range(self.num_domains) if d != test_domain]

            data_time.update(time.time() - end)

            objective = 0.
            optimizer.zero_grad()
            for p in self.model.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

            '''
            inner update phase
            '''
            inner_net = copy.deepcopy(self.model)
            # inner_net_params = [p for p in inner_net.parameters() if p.requires_grad]
            # outer_net_params = [p for p in self.model.parameters() if p.requires_grad]
            outer_opt_lr = self.get_lr(optimizer)
            inner_opt_lr = outer_opt_lr
            inner_optimizer = torch.optim.Adam(inner_net.module.get_params(), lr=inner_opt_lr, weight_decay=5e-4)
            train_inputs = [source_inputs[domain_id]['images'].cuda() for domain_id in train_domains]
            train_targets = [source_inputs[domain_id]['pids'].cuda() for domain_id in train_domains]

            inner_prob, inner_feat = inner_net(train_inputs)
            inner_prob = torch.cat(inner_prob, dim=0)
            inner_feat = torch.cat(inner_feat, dim=0)
            inner_targets = torch.cat(train_targets, dim=0)

            loss_inner_ce = self.criterion_ce(inner_prob, inner_targets)
            loss_inner_tri = self.criterion_tri(inner_feat, inner_targets)
            acc_inner = accuracy(inner_prob.view(-1, inner_prob.size(-1)).data, inner_targets.data)[0]
            loss_inner = loss_inner_ce + loss_inner_tri

            inner_optimizer.zero_grad()
            loss_inner.backward()
            inner_optimizer.step()

            # Now inner_net has accumulated gradients Gi
            # The clone-network (inner_net) has now parameters P - lr * Gi
            # Adding the gradient Gi to original network
            for p_outer, p_inner in zip(self.model.module.get_params(), inner_net.module.get_params()):
                if p_inner.grad is not None:
                    assert p_inner.grad.data.shape == p_outer.grad.data.shape
                    p_outer.grad.data.add_(p_inner.grad.data / self.num_domains)

            objective += loss_inner.item()

            '''
            outer update phase
            '''
            test_inputs = [source_inputs[test_domain]['images'].cuda()]
            test_targets = [source_inputs[test_domain]['pids'].cuda()]

            outer_prob, outer_feat = inner_net(test_inputs)

            outer_prob = torch.cat(outer_prob, dim=0)
            outer_feat = torch.cat(outer_feat, dim=0)
            test_targets = torch.cat(test_targets, dim=0)

            loss_outer_ce = self.criterion_ce(outer_prob, test_targets)
            loss_outer_tri = self.criterion_tri(outer_feat, test_targets)
            acc_outer = accuracy(outer_prob.view(-1, outer_prob.size(-1)).data, test_targets.data)[0]
            loss_outer = loss_outer_ce + loss_outer_tri

            grad_outer = torch.autograd.grad(loss_outer, inner_net.module.get_params(), allow_unused=True)

            objective += self.mldg_beta * loss_outer.item()

            for p_outer, g_outer in zip(self.model.module.get_params(), grad_outer):
                if g_outer is not None:
                    assert g_outer.data.shape == p_outer.grad.data.shape
                    p_outer.grad.data.add_(self.mldg_beta * g_outer.data / self.num_domains)

            for (n_outer, m_outer), (n_inner, m_inner) in zip(self.model.named_modules(), inner_net.named_modules()):
                assert n_outer == n_inner
                if isinstance(m_outer, nn.BatchNorm2d):
                    m_outer.running_mean.data = m_inner.running_mean.data.clone()
                    m_outer.running_var.data = m_inner.running_var.data.clone()
                    m_outer.num_batches_tracked.data = m_inner.num_batches_tracked.data.clone()
                if isinstance(m_outer, nn.BatchNorm1d):
                    m_outer.running_mean.data = m_inner.running_mean.data.clone()
                    m_outer.running_var.data = m_inner.running_var.data.clone()
                    m_outer.num_batches_tracked.data = m_inner.num_batches_tracked.data.clone()

            optimizer.step()

            # process inputs
            # inputs = [source['images'].cuda() for source in source_inputs]
            # targets = [source['pids'].cuda() for source in source_inputs]

            # t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            # device_num = torch.cuda.device_count()
            # B, C, H, W = s_inputs.size()
            #
            # def reshape(inputs):
            #     return inputs.view(device_num, -1, C, H, W)
            #
            # s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            # inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            # targets = targets.view(-1)
            # forward
            # prob, feats = self._forward(inputs)

            # loss_ce = 0.
            # loss_tri = 0.
            # acc = 0.

            # for domain_id, (cur_prob, cur_feat, target) in enumerate(zip(prob, feats, targets)):
            #     loss_ce += self.criterion_ce[domain_id](cur_prob, target)
            #     loss_tri += self.criterion_tri(cur_feat, target)
            #     acc += accuracy(cur_prob.view(-1, cur_prob.size(-1)).data, target.data)[0]

            # acc /= len(self.num_classes)

            # prob = prob[:, 0:source_classes + target_classes]

            # split feats
            # ori_feats = feats.view(device_num, -1, feats.size(-1))
            # feats_s, feats_t = ori_feats.split(ori_feats.size(1) // 2, dim=1)
            # ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            # loss_ce = self.criterion_ce(prob, targets)
            # loss_tri = self.criterion_tri(feats, targets)

            # enqueue and dequeue for xbm
            # if use_xbm:
            #     self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
            #     xbm_feats, xbm_targets = self.xbm.get()
            #     loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
            #     losses_xbm.update(loss_xbm.item())
            #     loss = loss_ce + loss_tri + loss_xbm
            # else:
            # loss = loss_ce + loss_tri

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # ori_prob = prob.view(device_num, -1, prob.size(-1))
            # prob_s, prob_t = ori_prob.split(ori_prob.size(1) // 2, dim=1)
            # prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()

            # prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(objective)
            losses_inner_ce.update(loss_inner_ce.item())
            losses_inner_tri.update(loss_inner_tri.item())
            losses_inner.update(loss_inner.item())
            losses_outer_ce.update(loss_outer_ce.item())
            losses_outer_tri.update(loss_outer_tri.item())
            losses_outer.update(loss_outer.item())
            precisions_inner.update(acc_inner.cpu().numpy()[0])
            precisions_outer.update(acc_outer.cpu().numpy()[0])
            precisions.update((acc_inner.cpu().numpy()[0] + acc_outer.cpu().numpy()[0]) / 2)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_inner {:.3f} ({:.3f}) '
                      'Loss_inner_ce {:.3f} ({:.3f}) '
                      'Loss_inner_tri {:.3f} ({:.3f}) '
                      'Loss_outer {:.3f} ({:.3f}) '
                      'Loss_outer_ce {:.3f} ({:.3f}) '
                      'Loss_outer_tri {:.3f} ({:.3f}) '
                      'Acc {:.2%} ({:.2%}) '
                      'Acc_inner {:.2%} ({:.2%}) '
                      'Acc_outer {:.2%} ({:.2%}) '
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_inner.val, losses_inner.avg,
                              losses_inner_ce.val, losses_inner_ce.avg,
                              losses_inner_tri.val, losses_inner_tri.avg,
                              losses_outer.val, losses_outer.avg,
                              losses_outer_ce.val, losses_outer_ce.avg,
                              losses_outer_tri.val, losses_outer_tri.avg,
                              precisions.val, precisions.avg,
                              precisions_inner.val, precisions_inner.avg,
                              precisions_outer.val, precisions_outer.avg,
                              ))

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def _parse_data(self, inputs):
        imgs = inputs['images']
        pids = inputs['pids']
        indexes = inputs['indices']
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.model(inputs)


class MDEGTrainer(object):
    def __init__(self, model, num_classes, num_domains=3, margin=None):
        super(MDEGTrainer, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.criterion_ce = CrossEntropyLabelSmooth(self.num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, optimizer, print_freq=50, train_iters=400):
        # self.criterion_ce = CrossEntropyLabelSmooth(source_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        # losses_xbm = AverageMeter()
        precisions = AverageMeter()
        # precisions_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = [loader.next() for loader in data_loader_source]
            # target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs = [source['images'].cuda() for source in source_inputs]
            targets = [source['pids'].cuda() for source in source_inputs]

            # t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            # device_num = torch.cuda.device_count()
            # B, C, H, W = s_inputs.size()
            #
            # def reshape(inputs):
            #     return inputs.view(device_num, -1, C, H, W)
            #
            # s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            # inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            # targets = targets.view(-1)
            # forward
            prob, feats = self._forward(inputs)
            prob = torch.cat(prob, dim=0)
            feats = torch.cat(feats, dim=0)
            targets = torch.cat(targets, dim=0)

            loss_tri = self.criterion_tri(feats, targets)
            loss_ce = self.criterion_ce(prob, targets)
            prec,  = accuracy(prob.view(-1, prob.size(-1)).data, targets.data)

            # prob = prob[:, 0:source_classes + target_classes]

            # split feats
            # ori_feats = feats.view(device_num, -1, feats.size(-1))
            # feats_s, feats_t = ori_feats.split(ori_feats.size(1) // 2, dim=1)
            # ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            # loss_ce = self.criterion_ce(prob, targets)
            # loss_tri = self.criterion_tri(feats, targets)

            # enqueue and dequeue for xbm
            # if use_xbm:
            #     self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
            #     xbm_feats, xbm_targets = self.xbm.get()
            #     loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
            #     losses_xbm.update(loss_xbm.item())
            #     loss = loss_ce + loss_tri + loss_xbm
            # else:
            loss = loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ori_prob = prob.view(device_num, -1, prob.size(-1))
            # prob_s, prob_t = ori_prob.split(ori_prob.size(1) // 2, dim=1)
            # prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()

            # prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(loss.item())
            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions.update(prec[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_ce {:.3f} ({:.3f}) '
                      'Loss_tri {:.3f} ({:.3f}) '
                      'Prec_s {:.2%} ({:.2%}) '
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg,
                              ))

    def _parse_data(self, inputs):
        imgs = inputs['images']
        pids = inputs['pids']
        indexes = inputs['indices']
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.model(inputs)


class MLDGSMMTrainer1(object):
    def __init__(self, model, num_classes, num_domains=3, margin=None, mldg_beta=0.5):
        super(MLDGSMMTrainer1, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.criterion_ce = CrossEntropyLabelSmooth(self.num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()
        self.mldg_beta = mldg_beta

    def train(self, epoch, data_loader_source, optimizer, print_freq=50, train_iters=400):
        # self.criterion_ce = CrossEntropyLabelSmooth(source_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_inner = AverageMeter()
        losses_inner_ce = AverageMeter()
        losses_inner_tri = AverageMeter()
        losses_outer = AverageMeter()
        losses_outer_ce = AverageMeter()
        losses_outer_tri = AverageMeter()
        precisions = AverageMeter()
        precisions_inner = AverageMeter()
        precisions_outer = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = [loader.next() for loader in data_loader_source]

            # test_domain = random.choice(range(self.num_domains))
            # train_domains = [d for d in range(self.num_domains) if d != test_domain]

            data_time.update(time.time() - end)

            objective = 0.
            optimizer.zero_grad()
            for p in self.model.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

            '''
            get meta-train and meta-test datasets
            '''
            test_inputs = [source_input['images'].cuda() for source_input in source_inputs]
            test_targets = [source_input['pids'].cuda() for source_input in source_inputs]

            train_inputs = self.get_mixed_inputs(test_inputs)
            train_targets = copy.deepcopy(test_targets)

            '''
            inner update phase
            '''
            inner_net = copy.deepcopy(self.model)
            # inner_net_params = [p for p in inner_net.parameters() if p.requires_grad]
            # outer_net_params = [p for p in self.model.parameters() if p.requires_grad]
            outer_opt_lr = self.get_lr(optimizer)
            inner_opt_lr = outer_opt_lr
            inner_optimizer = torch.optim.Adam(inner_net.module.get_params(), lr=inner_opt_lr, weight_decay=5e-4)

            inner_prob, inner_feat = inner_net(train_inputs)
            inner_prob = torch.cat(inner_prob, dim=0)
            inner_feat = torch.cat(inner_feat, dim=0)
            inner_targets = torch.cat(train_targets, dim=0)

            loss_inner_ce = self.criterion_ce(inner_prob, inner_targets)
            loss_inner_tri = self.criterion_tri(inner_feat, inner_targets)
            acc_inner = accuracy(inner_prob.view(-1, inner_prob.size(-1)).data, inner_targets.data)[0]
            loss_inner = loss_inner_ce + loss_inner_tri

            inner_optimizer.zero_grad()
            loss_inner.backward()
            inner_optimizer.step()

            # Now inner_net has accumulated gradients Gi
            # The clone-network (inner_net) has now parameters P - lr * Gi
            # Adding the gradient Gi to original network
            for p_outer, p_inner in zip(self.model.module.get_params(), inner_net.module.get_params()):
                if p_inner.grad is not None:
                    assert p_inner.grad.data.shape == p_outer.grad.data.shape
                    p_outer.grad.data.add_(p_inner.grad.data / self.num_domains)

            objective += loss_inner.item()

            '''
            outer update phase
            '''
            outer_prob, outer_feat = inner_net(test_inputs)

            outer_prob = torch.cat(outer_prob, dim=0)
            outer_feat = torch.cat(outer_feat, dim=0)
            test_targets = torch.cat(test_targets, dim=0)

            loss_outer_ce = self.criterion_ce(outer_prob, test_targets)
            loss_outer_tri = self.criterion_tri(outer_feat, test_targets)
            acc_outer = accuracy(outer_prob.view(-1, outer_prob.size(-1)).data, test_targets.data)[0]
            loss_outer = loss_outer_ce + loss_outer_tri

            grad_outer = torch.autograd.grad(loss_outer, inner_net.module.get_params(), allow_unused=True)

            objective += self.mldg_beta * loss_outer.item()

            for p_outer, g_outer in zip(self.model.module.get_params(), grad_outer):
                if g_outer is not None:
                    assert g_outer.data.shape == p_outer.grad.data.shape
                    p_outer.grad.data.add_(self.mldg_beta * g_outer.data / self.num_domains)

            for (n_outer, m_outer), (n_inner, m_inner) in zip(self.model.named_modules(), inner_net.named_modules()):
                assert n_outer == n_inner
                if isinstance(m_outer, nn.BatchNorm2d):
                    m_outer.running_mean.data = m_inner.running_mean.data.clone()
                    m_outer.running_var.data = m_inner.running_var.data.clone()
                    m_outer.num_batches_tracked.data = m_inner.num_batches_tracked.data.clone()
                if isinstance(m_outer, nn.BatchNorm1d):
                    m_outer.running_mean.data = m_inner.running_mean.data.clone()
                    m_outer.running_var.data = m_inner.running_var.data.clone()
                    m_outer.num_batches_tracked.data = m_inner.num_batches_tracked.data.clone()

            optimizer.step()

            # process inputs
            # inputs = [source['images'].cuda() for source in source_inputs]
            # targets = [source['pids'].cuda() for source in source_inputs]

            # t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            # device_num = torch.cuda.device_count()
            # B, C, H, W = s_inputs.size()
            #
            # def reshape(inputs):
            #     return inputs.view(device_num, -1, C, H, W)
            #
            # s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            # inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            # targets = targets.view(-1)
            # forward
            # prob, feats = self._forward(inputs)

            # loss_ce = 0.
            # loss_tri = 0.
            # acc = 0.

            # for domain_id, (cur_prob, cur_feat, target) in enumerate(zip(prob, feats, targets)):
            #     loss_ce += self.criterion_ce[domain_id](cur_prob, target)
            #     loss_tri += self.criterion_tri(cur_feat, target)
            #     acc += accuracy(cur_prob.view(-1, cur_prob.size(-1)).data, target.data)[0]

            # acc /= len(self.num_classes)

            # prob = prob[:, 0:source_classes + target_classes]

            # split feats
            # ori_feats = feats.view(device_num, -1, feats.size(-1))
            # feats_s, feats_t = ori_feats.split(ori_feats.size(1) // 2, dim=1)
            # ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            # loss_ce = self.criterion_ce(prob, targets)
            # loss_tri = self.criterion_tri(feats, targets)

            # enqueue and dequeue for xbm
            # if use_xbm:
            #     self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
            #     xbm_feats, xbm_targets = self.xbm.get()
            #     loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
            #     losses_xbm.update(loss_xbm.item())
            #     loss = loss_ce + loss_tri + loss_xbm
            # else:
            # loss = loss_ce + loss_tri

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # ori_prob = prob.view(device_num, -1, prob.size(-1))
            # prob_s, prob_t = ori_prob.split(ori_prob.size(1) // 2, dim=1)
            # prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()

            # prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(objective)
            losses_inner_ce.update(loss_inner_ce.item())
            losses_inner_tri.update(loss_inner_tri.item())
            losses_inner.update(loss_inner.item())
            losses_outer_ce.update(loss_outer_ce.item())
            losses_outer_tri.update(loss_outer_tri.item())
            losses_outer.update(loss_outer.item())
            precisions_inner.update(acc_inner.cpu().numpy()[0])
            precisions_outer.update(acc_outer.cpu().numpy()[0])
            precisions.update((acc_inner.cpu().numpy()[0] + acc_outer.cpu().numpy()[0]) / 2)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_inner {:.3f} ({:.3f}) '
                      'Loss_inner_ce {:.3f} ({:.3f}) '
                      'Loss_inner_tri {:.3f} ({:.3f}) '
                      'Loss_outer {:.3f} ({:.3f}) '
                      'Loss_outer_ce {:.3f} ({:.3f}) '
                      'Loss_outer_tri {:.3f} ({:.3f}) '
                      'Acc {:.2%} ({:.2%}) '
                      'Acc_inner {:.2%} ({:.2%}) '
                      'Acc_outer {:.2%} ({:.2%}) '
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_inner.val, losses_inner.avg,
                              losses_inner_ce.val, losses_inner_ce.avg,
                              losses_inner_tri.val, losses_inner_tri.avg,
                              losses_outer.val, losses_outer.avg,
                              losses_outer_ce.val, losses_outer_ce.avg,
                              losses_outer_tri.val, losses_outer_tri.avg,
                              precisions.val, precisions.avg,
                              precisions_inner.val, precisions_inner.avg,
                              precisions_outer.val, precisions_outer.avg,
                              ))

    @torch.no_grad()
    def get_mixed_inputs(self, all_inputs: list):
        num_domains = len(all_inputs)
        inputs = torch.cat(all_inputs, dim=0)

        batch_indices = torch.randperm(inputs.shape[0])
        style_inputs = inputs[batch_indices]
        lam = random.random()  # lam in range 0 to 1
        lam = lam * 0.5 + 0.5  # shrink to range 0.5 to 1
        mixed_inputs = adaptive_instance_normalization_v2(inputs, style_inputs, lam)

        mixed_inputs = torch.chunk(mixed_inputs, num_domains, dim=0)
        return mixed_inputs

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def _parse_data(self, inputs):
        imgs = inputs['images']
        pids = inputs['pids']
        indexes = inputs['indices']
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.model(inputs)


class MLDGSMMTrainer2(object):
    def __init__(self, model, num_classes, num_domains=3, margin=None, mldg_beta=0.5):
        super(MLDGSMMTrainer2, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.criterion_ce = CrossEntropyLabelSmooth(self.num_classes).cuda()
        self.criterion_tri = TripletLoss(margin=margin).cuda()
        self.mldg_beta = mldg_beta

    def train(self, epoch, data_loader_source, optimizer, print_freq=50, train_iters=400):
        # self.criterion_ce = CrossEntropyLabelSmooth(source_classes).cuda()

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_inner = AverageMeter()
        losses_inner_ce = AverageMeter()
        losses_inner_tri = AverageMeter()
        losses_outer = AverageMeter()
        losses_outer_ce = AverageMeter()
        losses_outer_tri = AverageMeter()
        precisions = AverageMeter()
        precisions_inner = AverageMeter()
        precisions_outer = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = [loader.next() for loader in data_loader_source]

            test_domain = random.choice(range(self.num_domains))
            train_domains = [d for d in range(self.num_domains) if d != test_domain]

            data_time.update(time.time() - end)

            objective = 0.
            optimizer.zero_grad()
            for p in self.model.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

            '''
            get meta-train and meta-test datasets
            '''
            train_inputs = [source_inputs[domain_id]['images'].cuda() for domain_id in train_domains]
            train_targets = [source_inputs[domain_id]['pids'].cuda() for domain_id in train_domains]
            test_inputs = [source_inputs[test_domain]['images'].cuda()]
            test_targets = [source_inputs[test_domain]['pids'].cuda()]

            train_inputs = self.get_mixed_inputs(train_inputs)

            '''
            inner update phase
            '''
            inner_net = copy.deepcopy(self.model)
            # inner_net_params = [p for p in inner_net.parameters() if p.requires_grad]
            # outer_net_params = [p for p in self.model.parameters() if p.requires_grad]
            outer_opt_lr = self.get_lr(optimizer)
            inner_opt_lr = outer_opt_lr
            inner_optimizer = torch.optim.Adam(inner_net.module.get_params(), lr=inner_opt_lr, weight_decay=5e-4)

            inner_prob, inner_feat = inner_net(train_inputs)
            inner_prob = torch.cat(inner_prob, dim=0)
            inner_feat = torch.cat(inner_feat, dim=0)
            inner_targets = torch.cat(train_targets, dim=0)

            loss_inner_ce = self.criterion_ce(inner_prob, inner_targets)
            loss_inner_tri = self.criterion_tri(inner_feat, inner_targets)
            acc_inner = accuracy(inner_prob.view(-1, inner_prob.size(-1)).data, inner_targets.data)[0]
            loss_inner = loss_inner_ce + loss_inner_tri

            inner_optimizer.zero_grad()
            loss_inner.backward()
            inner_optimizer.step()

            # Now inner_net has accumulated gradients Gi
            # The clone-network (inner_net) has now parameters P - lr * Gi
            # Adding the gradient Gi to original network
            for p_outer, p_inner in zip(self.model.module.get_params(), inner_net.module.get_params()):
                if p_inner.grad is not None:
                    assert p_inner.grad.data.shape == p_outer.grad.data.shape
                    p_outer.grad.data.add_(p_inner.grad.data / self.num_domains)

            objective += loss_inner.item()

            '''
            outer update phase
            '''
            outer_prob, outer_feat = inner_net(test_inputs)

            outer_prob = torch.cat(outer_prob, dim=0)
            outer_feat = torch.cat(outer_feat, dim=0)
            test_targets = torch.cat(test_targets, dim=0)

            loss_outer_ce = self.criterion_ce(outer_prob, test_targets)
            loss_outer_tri = self.criterion_tri(outer_feat, test_targets)
            acc_outer = accuracy(outer_prob.view(-1, outer_prob.size(-1)).data, test_targets.data)[0]
            loss_outer = loss_outer_ce + loss_outer_tri

            grad_outer = torch.autograd.grad(loss_outer, inner_net.module.get_params(), allow_unused=True)

            objective += self.mldg_beta * loss_outer.item()

            for p_outer, g_outer in zip(self.model.module.get_params(), grad_outer):
                if g_outer is not None:
                    assert g_outer.data.shape == p_outer.grad.data.shape
                    p_outer.grad.data.add_(self.mldg_beta * g_outer.data / self.num_domains)

            for (n_outer, m_outer), (n_inner, m_inner) in zip(self.model.named_modules(), inner_net.named_modules()):
                assert n_outer == n_inner
                if isinstance(m_outer, nn.BatchNorm2d):
                    m_outer.running_mean.data = m_inner.running_mean.data.clone()
                    m_outer.running_var.data = m_inner.running_var.data.clone()
                    m_outer.num_batches_tracked.data = m_inner.num_batches_tracked.data.clone()
                if isinstance(m_outer, nn.BatchNorm1d):
                    m_outer.running_mean.data = m_inner.running_mean.data.clone()
                    m_outer.running_var.data = m_inner.running_var.data.clone()
                    m_outer.num_batches_tracked.data = m_inner.num_batches_tracked.data.clone()

            optimizer.step()

            # process inputs
            # inputs = [source['images'].cuda() for source in source_inputs]
            # targets = [source['pids'].cuda() for source in source_inputs]

            # t_inputs, t_targets, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            # device_num = torch.cuda.device_count()
            # B, C, H, W = s_inputs.size()
            #
            # def reshape(inputs):
            #     return inputs.view(device_num, -1, C, H, W)
            #
            # s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            # inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # targets = torch.cat((s_targets.view(device_num, -1), t_targets.view(device_num, -1)), 1)
            # targets = targets.view(-1)
            # forward
            # prob, feats = self._forward(inputs)

            # loss_ce = 0.
            # loss_tri = 0.
            # acc = 0.

            # for domain_id, (cur_prob, cur_feat, target) in enumerate(zip(prob, feats, targets)):
            #     loss_ce += self.criterion_ce[domain_id](cur_prob, target)
            #     loss_tri += self.criterion_tri(cur_feat, target)
            #     acc += accuracy(cur_prob.view(-1, cur_prob.size(-1)).data, target.data)[0]

            # acc /= len(self.num_classes)

            # prob = prob[:, 0:source_classes + target_classes]

            # split feats
            # ori_feats = feats.view(device_num, -1, feats.size(-1))
            # feats_s, feats_t = ori_feats.split(ori_feats.size(1) // 2, dim=1)
            # ori_feats = torch.cat((feats_s, feats_t), 1).view(-1, ori_feats.size(-1))

            # classification+triplet
            # loss_ce = self.criterion_ce(prob, targets)
            # loss_tri = self.criterion_tri(feats, targets)

            # enqueue and dequeue for xbm
            # if use_xbm:
            #     self.xbm.enqueue_dequeue(ori_feats.detach(), targets.detach())
            #     xbm_feats, xbm_targets = self.xbm.get()
            #     loss_xbm = self.criterion_tri_xbm(ori_feats, targets, xbm_feats, xbm_targets)
            #     losses_xbm.update(loss_xbm.item())
            #     loss = loss_ce + loss_tri + loss_xbm
            # else:
            # loss = loss_ce + loss_tri

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # ori_prob = prob.view(device_num, -1, prob.size(-1))
            # prob_s, prob_t = ori_prob.split(ori_prob.size(1) // 2, dim=1)
            # prob_s, prob_t = prob_s.contiguous(), prob_t.contiguous()

            # prec_t, = accuracy(prob_t.view(-1, prob_s.size(-1)).data, t_targets.data)

            losses.update(objective)
            losses_inner_ce.update(loss_inner_ce.item())
            losses_inner_tri.update(loss_inner_tri.item())
            losses_inner.update(loss_inner.item())
            losses_outer_ce.update(loss_outer_ce.item())
            losses_outer_tri.update(loss_outer_tri.item())
            losses_outer.update(loss_outer.item())
            precisions_inner.update(acc_inner.cpu().numpy()[0])
            precisions_outer.update(acc_outer.cpu().numpy()[0])
            precisions.update((acc_inner.cpu().numpy()[0] + acc_outer.cpu().numpy()[0]) / 2)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f}) '
                      'Data {:.3f} ({:.3f}) '
                      'Loss {:.3f} ({:.3f}) '
                      'Loss_inner {:.3f} ({:.3f}) '
                      'Loss_inner_ce {:.3f} ({:.3f}) '
                      'Loss_inner_tri {:.3f} ({:.3f}) '
                      'Loss_outer {:.3f} ({:.3f}) '
                      'Loss_outer_ce {:.3f} ({:.3f}) '
                      'Loss_outer_tri {:.3f} ({:.3f}) '
                      'Acc {:.2%} ({:.2%}) '
                      'Acc_inner {:.2%} ({:.2%}) '
                      'Acc_outer {:.2%} ({:.2%}) '
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_inner.val, losses_inner.avg,
                              losses_inner_ce.val, losses_inner_ce.avg,
                              losses_inner_tri.val, losses_inner_tri.avg,
                              losses_outer.val, losses_outer.avg,
                              losses_outer_ce.val, losses_outer_ce.avg,
                              losses_outer_tri.val, losses_outer_tri.avg,
                              precisions.val, precisions.avg,
                              precisions_inner.val, precisions_inner.avg,
                              precisions_outer.val, precisions_outer.avg,
                              ))

    @torch.no_grad()
    def get_mixed_inputs(self, all_inputs: list):
        num_domains = len(all_inputs)
        inputs = torch.cat(all_inputs, dim=0)

        batch_indices = torch.randperm(inputs.shape[0])
        style_inputs = inputs[batch_indices]
        lam = random.random()  # lam in range 0 to 1
        lam = lam * 0.5 + 0.5  # shrink to range 0.5 to 1
        mixed_inputs = adaptive_instance_normalization_v2(inputs, style_inputs, lam)

        mixed_inputs = torch.chunk(mixed_inputs, num_domains, dim=0)
        return mixed_inputs

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def _parse_data(self, inputs):
        imgs = inputs['images']
        pids = inputs['pids']
        indexes = inputs['indices']
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.model(inputs)
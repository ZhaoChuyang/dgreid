from __future__ import print_function, absolute_import

import random
import time
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


class SMMTrainer_v2(object):
    def __init__(self, model, num_classes, margin=None):
        super(SMMTrainer_v2, self).__init__()
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

import os
import time
import torch
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torchvision
from torchsummary import summary
from torch.autograd import Variable
import os
import argparse
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import _LRScheduler
from lib.mca import mca
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
from ptflops import get_model_complexity_info

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def bce_iou_loss(pred, mask):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)

    weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    weighted_iou = (weight * iou).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    return (weighted_bce + weighted_iou).mean()


def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
            # ---- loss function ----
            loss4 = structure_loss(lateral_map_4, gts)
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)

            loss = loss2 + loss3 + loss4  # TODO: try different weights for loss
            # loss = loss2
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
            if epoch < 5:
                warmup_scheduler.step()
                warm_lr = warmup_scheduler.get_lr()
                # print("warm_lr:%s" % warm_lr)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record2.show(), loss_record3.show(), loss_record4.show()))
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'MCANet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'MCANet-%d.pth'% epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=5e-5, help='learning rate')
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer Adam or SGD')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=20, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--augmentation',
                        default='True', help='choose to do random flip rotation')
    parser.add_argument('--test_path', type=str,
                        default='./data/TestDataset/{}', help='path to testing Kvasir dataset')
    parser.add_argument('--train_save', type=str,
                        default='mca')
    opt = parser.parse_args()
    # ---- build models ----
    model = mca().cuda()
    macs, params = get_model_complexity_info(model, (3, 352, 352), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    summary(model, (3, 352, 352))

    params = model.parameters()
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, momentum=0.99, weight_decay=5e-4)
        # optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)
    # optimizer = torch.optim.Adam(params, opt.lr)
    print(optimizer)
    warmup_epoch = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100 - warmup_epoch)

    image_root = '{}/image/'.format(opt.train_path)
    gt_root = '{}/mask/'.format(opt.train_path)
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    # iter_per_epoch = len(train_loader)
    total_step = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, total_step * warmup_epoch)


    print("#" * 20, "Start Training", "#" * 20)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optimizer.zero_grad()
    optimizer.step()
    for epoch in range(1, opt.epoch):
        if epoch >= warmup_epoch:
            scheduler.step()
            learn_rate = scheduler.get_lr()[0]
	    train(train_loader, model, optimizer, epoch, opt.test_path)


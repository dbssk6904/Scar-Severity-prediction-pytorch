import argparse
import os
import shutil
import time
import random
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn.model_selection import StratifiedKFold

import torch.onnx

from MODELS.model_resnet import *
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet', help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--depth', default=50, type=int, metavar='D', help='model depth')
parser.add_argument('--ngpu', default=4, type=int, metavar='G', help='number of gpus to use')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument("--prefix", type=str, required=True, metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluation only')
parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default=None)
parser.add_argument('--kfold', type=int, default=10, metavar='K')
best_prec1 = 0

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')


def main():
    global args, best_prec1
    global viz, train_lot, test_lot
    args = parser.parse_args()
    print("args", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')    # ./data/train
    valdir = os.path.join(args.data, 'val')        # ./data/val

    normalize = transforms.Normalize(mean=[0.5959581200688205, 0.46351973281645936, 0.4014567226013591],
                                     std=[0.07559669492871386, 0.0801965185805582, 0.08250758011366909])

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([transforms.Resize((224, 224)),
                                                                       transforms.ToTensor(),
                                                                       normalize]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize((224, 224)),
                                                                    transforms.ToTensor(),
                                                                    normalize]))

    # Stratified k-fold cross-validation
    dataset = ConcatDataset([train_dataset, val_dataset])
    train_y = train_dataset.targets
    valid_y = val_dataset.targets
    Y = ConcatDataset([train_y, valid_y])

    splits = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    foldperf={}

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)), Y)):
        print('Fold {}', format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.workers, pin_memory=True)

        # create model
        if args.arch == "resnet":
            model = ResidualNet('ImageNet', args.depth, 4, args.att_type)

        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        model = model.cuda()

        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

        history = {'valid_loss': [], 'valid_acc': [], 'train_idx': [], 'val_idx': []}
        history['train_idx'].append(train_idx)
        history['val_idx'].append(val_idx)

        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            prec1, mloss = validate(val_loader, model, criterion, epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.prefix, args.lr, args.momentum, args.weight_decay)

            history['valid_loss'].append(mloss)
            history['valid_acc'].append(prec1)

        foldperf['fold{}'.format(fold+1)] = history

    with open("foldperf_image-based-model.pkl", "wb") as f:
        pickle.dump(foldperf, f)

    valid_acc_f, valid_loss_f = [], []
    for f in range(1, args.kfold+1):
        acc_list = foldperf['fold{}'.format(f)]['valid_acc']
        valid_acc_f.append(sum(acc_list) / len(acc_list))
        loss_list = foldperf['fold{}'.format(f)]['valid_loss']
        valid_loss_f.append(sum(loss_list) / len(loss_list))

    print('Performance of {} fold cross validation'.format(args.kfold))
    print("Average Average Valid Loss: {:.3f} \t Average Valid Acc: {:.3f}".format((sum(valid_loss_f)/len(valid_loss_f)), (sum(valid_acc_f)/len(valid_acc_f))))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec2 = accuracy(output.data, target, topk=(1, 2))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top2.update(prec2[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@4 {top2.val:.3f} ({top2.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top2=top2))


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec2 = accuracy(output.data, target, topk=(1, 2))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top2.update(prec2[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@4 {top2.val:.3f} ({top2.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top2=top2))

    print(' * Prec@1 {top1.avg:.3f} Prec@4 {top2.avg:.3f} *Loss-{loss.avg:.3f}'
          .format(top1=top1, top2=top2, loss=losses))

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, prefix, lr, momentum, weight_decay):
    filename = './checkpoints/%s_checkpoint.pth.tar' % prefix
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/%.3f_%.1f_%.5f_image_best.pth.tar' % (lr, momentum, weight_decay))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

import argparse
import os
import shutil
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import torch.onnx

from MODELS.model_mlp import *


parser = argparse.ArgumentParser(description='Clinical-data-based Evaluating')

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--ngpu', default=2, type=int, metavar='G', help='number of gpus to use')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument("--prefix", type=str, required=True, metavar='PFX', help='prefix for logging & checkpoint saving')
best_prec1 = 0

if not os.path.exists('./clinical_output'):
    os.mkdir('./clinical_output')


def main():
    global args, best_prec1
    global viz, train_lot, test_lot
    args = parser.parse_args()
    print("args", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # create model
    model = MLP(4)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    model = model.cuda()
    print("model")
    print(model)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    test_df = pd.read_excel(args.data + '/thyroid_220911_Px_test.xlsx', header=0)

    X_test = test_df.drop(['img_name', 'Cls'], axis=1)
    Y_test = test_df['Cls']

    testsets = TensorData(X_test.values, Y_test.values)
    test_loader = torch.utils.data.DataLoader(testsets, batch_size=args.batch_size, shuffle=False)
    validate(test_loader, model, criterion, 0)


def validate(valid_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(valid_loader):
        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(inputs)
            target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        torch.save(output, './clinical_output/output_tensor_{}.pt'.format(i))
        torch.save(target, './clinical_output/target_tensor_{}.pt'.format(i))

        prec1, prec2 = accuracy(output.data, target, topk=(1, 2))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top2.update(prec2[0], inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@4 {top2.val:.3f} ({top2.avg:.3f})'.format(
                i, len(valid_loader), loss=losses, top1=top1, top2=top2))

    return top1.avg, losses.avg



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


class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    main()













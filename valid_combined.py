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
import torchvision.transforms as transforms

import torch.onnx

from MODELS.concatenate import *
from PIL import Image
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
parser = argparse.ArgumentParser(description='Combined-model Evaluating')


parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--depth', default=50, type=int, metavar='D', help='model depth')
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
parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default=None)
best_prec1 = 0

if not os.path.exists('./combined_output'):
    os.mkdir('./combined_output')


def main():
    global args, best_prec1
    global viz, train_lot, test_lot
    args = parser.parse_args()
    print("args", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # create model
    model = TwoInputNet(4, args.depth, args.att_type)

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
    test_img_dir = os.path.join(args.data, 'test_all')
    normalize = transforms.Normalize(mean=[0.5959581200688205, 0.46351973281645936, 0.4014567226013591],
                                     std=[0.07559669492871386, 0.0801965185805582, 0.08250758011366909])
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    # test_df have to scaled using 'scaler.fit(best_fold_train_df)'
    test_df = pd.read_excel(args.data + '/thyroid_220911_Px_test.xlsx', header=0)

    test_dataset = CombineDataset(test_df, 'img_name', 'Cls', test_img_dir, transform=tf)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    validate(test_loader, model, criterion, 0)


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (img_input, ft_input, target) in enumerate(val_loader):
        target = target.cuda()
        with torch.no_grad():
            ft_input_var = torch.autograd.Variable(ft_input)
            img_input_var = torch.autograd.Variable(img_input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(img_input_var, ft_input_var)
        loss = criterion(output, target_var)

        torch.save(output, './combined_output/output_tensor_{}.pt'.format(i))
        torch.save(target, './combined_output/target_tensor_{}.pt'.format(i))

        # measure accuracy and record loss
        prec1, prec2 = accuracy(output.data, target, topk=(1, 2))
        losses.update(loss.item(), ft_input.size(0))
        top1.update(prec1[0], ft_input.size(0))
        top2.update(prec2[0], ft_input.size(0))

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

    print(' * Prec@1 {top1.avg:.3f} Prec@4 {top2.avg:.3f} *Loss {loss.avg:.3f}'
          .format(top1=top1, top2=top2, loss=losses))

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


class CombineDataset(Dataset):
    def __init__(self, frame, id_col, label_name, path_imgs, transform=None):
        """
        Args:
            frame (pd.DataFrame): Frame with the tabular data.
            id_col (string): Name of the column that connects image to tabular data
            label_name (string): Name of the column with the label to be predicted
            path_imgs (string): path to the folder where the images are.
            transform (callable, optional): Optional transform to be applied
                on a sample, you need to implement a transform to use this.
        """
        self.frame = frame
        self.id_col = id_col
        self.label_name = label_name
        self.path_imgs = path_imgs
        self.transform = transform

    def __len__(self):
        return (self.frame.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get images
        img_name = self.frame[self.id_col].iloc[idx]
        path = os.path.join(self.path_imgs, img_name)
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)

        # get features
        feats = [feat for feat in self.frame.columns if feat not in [self.label_name, self.id_col]]
        feats = np.array(self.frame[feats].iloc[idx])
        feats = torch.from_numpy(feats.astype(np.float32))

        # get label
        label = np.array(self.frame[self.label_name].iloc[idx])
        label = torch.from_numpy(label.astype(np.int64))

        return image, feats, label


if __name__ == '__main__':
    main()








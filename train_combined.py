import argparse
import os
import shutil
import time
import random
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import torch.onnx

from MODELS.concatenate import *
from PIL import Image
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
parser = argparse.ArgumentParser(description='Combined-model Training')

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
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument("--prefix", type=str, required=True, metavar='PFX', help='prefix for logging & checkpoint saving')
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
    train_img_dir = os.path.join(args.data, 'train')
    val_img_dir = os.path.join(args.data, 'val')
    concat_img_dir = os.path.join(args.data, 'concat_all')

    normalize = transforms.Normalize(mean=[0.5959581200688205, 0.46351973281645936, 0.4014567226013591],
                                     std=[0.07559669492871386, 0.0801965185805582, 0.08250758011366909])

    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    train_df = pd.read_excel(args.data + '/thyroid_220911_Px_train_aug.xlsx', header=0)
    val_df = pd.read_excel(args.data + '/thyroid_220911_Px_val.xlsx', header=0)
    concat_df = pd.read_excel(args.data + '/thyroid_220911_Px_concatset.xlsx', header=0)

    train_dataset = CombineDataset(train_df, 'img_name', 'Cls', train_img_dir, transform=tf)
    val_dataset = CombineDataset(val_df, 'img_name', 'Cls', val_img_dir, transform=tf)

    # Stratified k-fold cross-validation
    dataset = ConcatDataset([train_dataset, val_dataset])
    train_y = train_dataset.frame['Cls']
    valid_y = val_dataset.frame['Cls']
    Y = ConcatDataset([train_y, valid_y])

    splits = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    scaler = StandardScaler()
    foldperf={}

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)), Y)):
        print('** Fold {} **'.format(fold + 1))

        scaled_train_df, scaled_valid_df = scaled_datasets(concat_df, train_idx, val_idx, scaler)
        fold_train_dataset = CombineDataset(scaled_train_df, 'img_name', 'Cls', concat_img_dir, transform=tf)
        fold_val_dataset = CombineDataset(scaled_valid_df, 'img_name', 'Cls', concat_img_dir, transform=tf)

        train_loader = DataLoader(fold_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(fold_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

        # create model
        model = TwoInputNet(4, args.depth, args.att_type)

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
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.prefix, args.lr, args.momentum, args.weight_decay)

            history['valid_loss'].append(mloss)
            history['valid_acc'].append(prec1)

        foldperf['fold{}'.format(fold+1)] = history

    with open("foldperf_combined-model.pkl", "wb") as f:
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
    for i, (img_input, ft_input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        ft_input_var = torch.autograd.Variable(ft_input)
        img_input_var = torch.autograd.Variable(img_input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(img_input_var, ft_input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec2 = accuracy(output.data, target, topk=(1, 2))
        losses.update(loss.item(), ft_input.size(0))
        top1.update(prec1[0], ft_input.size(0))
        top2.update(prec2[0], ft_input.size(0))

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
    for i, (img_input, ft_input, target) in enumerate(val_loader):
        target = target.cuda()
        with torch.no_grad():
            ft_input_var = torch.autograd.Variable(ft_input)
            img_input_var = torch.autograd.Variable(img_input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(img_input_var, ft_input_var)
        loss = criterion(output, target_var)

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


def save_checkpoint(state, is_best, prefix, lr, momentum, weight_decay):
    filename = './checkpoints/%s_checkpoint.pth.tar' % prefix
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/%.3f_%.1f_%.5f_combined_best.pth.tar' % (lr, momentum, weight_decay))


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
        path = os.path.join(self.path_imgs, img_name)              # train_dataset = CombineDataset(train_df, 'img_name', 'Cls', train_img_dir, transform=tf)
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

def scaled_datasets(df, train_idx, val_idx, scaler):
    train = df.loc[train_idx].reset_index(drop=True)
    valid = df.loc[val_idx].reset_index(drop=True)

    scaled_train, scaled_valid = scaling(train, valid, scaler)   # scaler = StandardScaler()

    scaled_train_df = pd.concat([train, scaled_train], axis=1)
    scaled_train_df = scaled_train_df.drop(['age', 'BMI', 'Delta_date'], axis=1)

    scaled_valid_df = pd.concat([valid, scaled_valid], axis=1)
    scaled_valid_df = scaled_valid_df.drop(['age', 'BMI', 'Delta_date'], axis=1)

    return scaled_train_df, scaled_valid_df


def scaling(train, valid, scaler):
    age_train = train[['age']]
    bmi_train = train[['BMI']]
    date_train = train[['Delta_date']]

    age_valid = valid[['age']]
    bmi_valid = valid[['BMI']]
    date_valid = valid[['Delta_date']]

    age_scaler = scaler.fit(age_train)
    bmi_scaler = scaler.fit(bmi_train)
    date_scaler = scaler.fit(date_train)

    label_1 = age_scaler.fit_transform(age_train)
    label_2 = bmi_scaler.fit_transform(bmi_train)
    label_3 = date_scaler.fit_transform(date_train)

    label_4 = age_scaler.fit_transform(age_valid)
    label_5 = bmi_scaler.fit_transform(bmi_valid)
    label_6 = date_scaler.fit_transform(date_valid)

    scaled_train = get_scaled_df(label_1, label_2, label_3)
    scaled_valid = get_scaled_df(label_4, label_5, label_6)
    return scaled_train, scaled_valid


def get_scaled_df(value_1, value_2, value_3):
    values = {'0':[], '1':[], '2':[]}
    value_list = [value_1, value_2, value_3]
    scaled_df = pd.DataFrame()

    for j, f in enumerate(value_list):
        for i in range(len(f)):
            value = f[i][0]
            if str(value) == 'nan':
                values[str(j)].append(0.0)
                continue
            values[str(j)].append(value)
    scaled_df['scaled_age'] = values['0']
    scaled_df['scaled_BMI'] = values['1']
    scaled_df['scaled_Delta_date'] = values['2']
    return scaled_df

if __name__ == '__main__':
    main()

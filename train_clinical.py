import argparse
import os
import shutil
import time
import random
import pandas as pd

from torch.utils.data import DataLoader

from MODELS.model_mlp import *



parser = argparse.ArgumentParser(description='PyTorch Multi Input Layer model Training')

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--ngpu', default=2, type=int, metavar='G', help='number of gpus to use')
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

    model = MLP(4)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Data loading code
    train_df = pd.read_excel(args.data, '/THYROID_220911_Px_train_aug.xlsx', header=0)
    val_df = pd.read_excel(args.data, '/THYROID_220911_Px_val.xlsx', header=0)

    X_train = train_df.drop(['img_name', 'Cls'], axis=1)
    Y_train = train_df['Cls']

    X_valid = val_df.drop(['img_name', 'Cls'], axis=1)
    Y_valid = val_df['Cls']

    trainsets = TensorData(X_train.values, Y_train.values)
    validsets = TensorData(X_valid.values, Y_valid.values)

    train_loader = torch.utils.data.DataLoader(trainsets, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validsets, batch_size=args.batch_size, shuffle=False)


    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, mloss = validate(valid_loader, model, criterion, epoch)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.prefix, args.lr, args.momentum, args.weight_decay)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    for i, (inputs, target) in enumerate(train_loader):  # 무작위로 섞인 16개의 데이터가 담긴 배치가 하나씩 들어옴
        target = target.cuda()
        input_var = torch.autograd.Variable(inputs)
        target_var = torch.autograd.Variable(target)

        outputs = model(input_var)
        loss = criterion(outputs, target_var)

        prec1, prec2 = accuracy(outputs.data, target, topk=(1, 2))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top2.update(prec2[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@4 {top2.val:.3f} ({top2.avg:.3f})'.format(
                epoch, i, len(train_loader), loss=losses, top1=top1, top2=top2))


def validate(valid_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    model.eval()

    for i, (inputs, target) in enumerate(valid_loader):
        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(inputs)
            target_var = torch.autograd.Variable(target)

        outputs = model(input_var)
        loss = criterion(outputs, target_var)

        prec1, prec2 = accuracy(outputs.data, target, topk=(1, 2))
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


def save_checkpoint(state, is_best, prefix, lr, momentum, weight_decay):
    filename = './checkpoints/%s_checkpoint.pth.tar' % prefix
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/%.3f_%.1f_%.5f_clinical_best.pth.tar' % (lr, momentum, weight_decay))


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













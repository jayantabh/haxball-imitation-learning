import argparse
import time
import copy

import numpy as np
import torch
import torchvision


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from replays_dataset import HaxballDemoDataset

from .models import MyModel

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

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct / batch_size

    return acc

def train(epoch, data_loader, model, optimizer, criterion):

    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    for idx, (data, target) in enumerate(data_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        #############################################################################
        # TODO: Complete the body of training loop                                  #
        #       1. forward data batch to the model                                  #
        #       2. Compute batch loss                                               #
        #       3. Compute gradients and update model parameters                    #
        #############################################################################
        optimizer.zero_grad()
        out = model(data)

        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        batch_acc = accuracy(out, target)

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
                   .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses, top1=acc))


def validate(epoch, val_loader, model, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    num_class = 10
    cm = torch.zeros(num_class, num_class)
    # evaluation loop
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        #############################################################################
        # TODO: Complete the body of training loop                                  #
        #       HINT: torch.no_grad()                                               #
        #############################################################################
        with torch.no_grad():
            out = model(data)
            loss = criterion(out, target)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        batch_acc = accuracy(out, target)

        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
               'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t')
               .format(epoch, idx, len(val_loader), iter_time=iter_time, loss=losses, top1=acc))
    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
    return acc.avg, cm


def adjust_learning_rate(optimizer, epoch, args):
    epoch += 1
    if epoch <= args.warmup:
        lr = args.learning_rate * epoch / args.warmup
    elif epoch > args.steps[1]:
        lr = args.learning_rate * 0.01
    elif epoch > args.steps[0]:
        lr = args.learning_rate * 0.1
    else:
        lr = args.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():

    dataset = replays_dataset("sample_preprocessed")

    train_dataset, test_dataset = random_split(dataset, [len(dataset) * .8, 1 - len(dataset)*.8])

    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False)

    model = MyModel()
    print(model)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters())
    best = 0.0
    best_cm = None
    best_model = None

    epochs = 10
    for epoch in range(epochs):
        # train loop
        train(epoch, train_loader, model, optimizer, criterion)

        # validation loop
        acc, cm = validate(epoch, test_loader, model, criterion)

        if acc > best:
            best = acc
            best_cm = cm
            best_model = copy.deepcopy(model)

    print('Best Prec @1 Acccuracy: {:.4f}'.format(best))
    per_cls_acc = best_cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    torch.save(best_model.state_dict(), './checkpoints/model.pth')


if __name__ == '__main__':
    main()
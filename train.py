import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, dataset, random_split

from tqdm import tqdm

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

def accuracy(output, target, num_actions):
    """Computes the precision@k for the specified values of k"""
    num_actions *= 1.0
    batch_size = target.shape[0]

    correct = output.eq(target).sum() * 1.0

    acc = correct / (batch_size * num_actions)

    return acc

def train(epoch, data_loader, model, optimizer, criterion):

    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    for data, target in tqdm(data_loader):

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

        batch_acc = accuracy(out, target, len(target))

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

def validate(epoch, val_loader, model, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # evaluation loop
    for data, target in val_loader:

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

        out = (out > 0.5).type(torch.int)

        batch_acc = accuracy(out, target, len(target))

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

    print(f"Epoch: {epoch}, Accuracy: {acc.avg.item():.2f}")
    return acc.avg


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

def main(model, dataset, saved_model_path):
    train_dataset, test_dataset = random_split(dataset, [len(dataset) - len(dataset)//10, len(dataset)//10])

    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)

    print(model)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.BCELoss()

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best = 0.0
    best_model = None

    epochs = 10
    for epoch in range(epochs):
        # train loop
        train(epoch, train_loader, model, optimizer, criterion)

        # validation loop
        acc = validate(epoch, test_loader, model, criterion)

        if acc > best:
            best = acc
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), saved_model_path)

    print('Best Prec Acccuracy: {:.4f}'.format(best))


import argparse
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=0, help="Must specify --dataset type")
    parser.add_argument("--model", type=str, help="Must specify --model")
    parser.add_argument("--name", type=str, help="Must specify saved model name")
    args = parser.parse_args()
	
    saved_model_path = os.path.join( os.getcwd(), 'saved_models', args.name)

    if args.model == "Basic":
        from models.BasicModel import BasicModel 
        model = BasicModel()        
    elif args.model == "Dist":
        from models.DistModel import DistModel
        model = DistModel()
    elif args.model == "Basic3v3":
        from models.Basic3v3 import Basic3v3
        model = Basic3v3()

    if args.dataset == "BasicFiltered":
        from datasets.BasicFilteredDataset import BasicFilteredDataset
        my_dataset = BasicFilteredDataset("sample_preprocessed")
        
    elif args.dataset == "DistBot":
        from datasets.DistDataset import DistDataset
        my_dataset = DistDataset("sample_preprocessed")
    
    elif args.dataset == "Basic3v3":
        from datasets.Basic3v3 import Basic3v3
        my_dataset = Basic3v3("sample_preprocessed")

    main(model, my_dataset, saved_model_path)
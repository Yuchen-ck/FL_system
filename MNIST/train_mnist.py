import argparse
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# from mnist_net import mnist_net
from mnist_net import *

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='train_output.log',
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)


def get_args():

    #在這裡調參:
    batch_size = 128 # batch size設16，收斂時間會拉長。
    epochs = 10 
    lr_max = 5e-3  #0.005
    lr_type = 'cyclic'  #cyclic learning rate 效果比較好

    # FGSM: epsilon + alpha 
    attack = 'none'
    epsilon = 0.3
    alpha = 0.375

    # PGD: epsilon + alpha + attack-iters
    # attack = 'pgd'
    # epsilon = 0.3
    # alpha = 0.375
    attack_iters = 40


    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=batch_size, type=int)
    parser.add_argument('--epochs', default = epochs, type=int)
    parser.add_argument('--attack', default = attack, type=str, choices=['none', 'pgd', 'fgsm'])
    parser.add_argument('--epsilon', default = epsilon, type=float)
    parser.add_argument('--alpha', default = alpha, type=float)
    parser.add_argument('--attack-iters', default = attack_iters, type=int)
    parser.add_argument('--lr-max', default = lr_max, type=float)
    parser.add_argument('--lr-type', default = lr_type, choices=['cyclic', 'flat'])
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    mnist_train = datasets.MNIST("./mnist-data", train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)

    model = Net().cuda()
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr_max)
    
    if args.lr_type == 'cyclic': 
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2//5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_type == 'flat': 
        lr_schedule = lambda t: args.lr_max
    else:
        raise ValueError('Unknown lr_type')

    criterion = nn.CrossEntropyLoss()

    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            lr = lr_schedule(epoch + (i+1)/len(train_loader))
            opt.param_groups[0].update(lr=lr)

            if args.attack == 'fgsm':
                #這個好像是關鍵!!!
                delta = torch.zeros_like(X).uniform_(-args.epsilon, args.epsilon).cuda() #在[-args.epsilon, args.epsilon]的隨機均匀分布中取值，並重新赋值
                delta.requires_grad = True
                output = model(X + delta)
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                delta.data = torch.clamp(delta + args.alpha * torch.sign(grad), -args.epsilon, args.epsilon)
                delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
                delta = delta.detach()

            elif args.attack == 'none':
                delta = torch.zeros_like(X)

            elif args.attack == 'pgd':
                delta = torch.zeros_like(X).uniform_(-args.epsilon, args.epsilon)
                delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
                for _ in range(args.attack_iters):
                    delta.requires_grad = True
                    output = model(X + delta)
                    loss = criterion(output, y)
                    opt.zero_grad()
                    loss.backward()
                    grad = delta.grad.detach()
                    I = output.max(1)[1] == y
                    delta.data[I] = torch.clamp(delta + args.alpha * torch.sign(grad), -args.epsilon, args.epsilon)[I]
                    delta.data[I] = torch.max(torch.min(1-X, delta.data), 0-X)[I]
                delta = delta.detach()
            
            output = model(torch.clamp(X + delta, 0, 1))
            loss = criterion(output, y)
            
            opt.zero_grad()
            loss.backward() #Fast Adversarail Training: backward兩次 
            opt.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        train_time = time.time()
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
            epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)
        # torch.save(model.state_dict(), "pgd_attack.pt")
        torch.save(model.state_dict(), "clean.pt")
    
    print("對抗式訓練結束")


if __name__ == "__main__":
    main()

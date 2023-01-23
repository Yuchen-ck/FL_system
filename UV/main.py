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
from sklearn.metrics import r2_score
# from mnist_net import mnist_net

from uv_model import *
from preprocessing import *
from attack import *

logger = logging.getLogger(__name__)

logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    filename = "train_output.log",
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    attack = 'pgd' #['none', 'pgd', 'fgsm']
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=1440, type=int)
    parser.add_argument('--epochs', default=1000, type=int) #epoch拉長
    parser.add_argument('--attack', default = attack, type=str, choices=['none', 'pgd', 'fgsm'])
    parser.add_argument('--epsilon', default=0.4, type=float)
    parser.add_argument('--alpha', default=0.375, type=float)
    parser.add_argument('--attack-iters', default=40, type=int)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-type', default='cyclic') #cyclic flat
    
    parser.add_argument('--modelPath', default= attack+"_defense.pt", type=str)
    return parser.parse_args()


def main():
    args = get_args()
    logger.info(args)

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    

    # mnist_train = datasets.MNIST("../mnist-data", train=True, download=True, transform=transforms.ToTensor())
    # train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)
    batch_sz = args.batch_size
    train_loader = load_train_dataloader(batch_sz)


    model = Model().to(device)
    # model.load_state_dict(torch.load('fgsm_defense.pt'))
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr_max)
   
    if args.lr_type == 'cyclic': 
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2//5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_type == 'flat': 
        lr_schedule = lambda t: args.lr_max
    else:
        raise ValueError('Unknown lr_type')

    criterion = nn.MSELoss()

    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train R2_score')
    print('Epoch \t Time \t LR \t \t Train Loss \t Train R2_score')
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = 0
        train_n = 0
        test_r2_score = []
        for i,(X, y,index) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            lr = lr_schedule(epoch + (i+1)/len(train_loader))
            opt.param_groups[0].update(lr=lr)

            # Adv_x = half_adversarial(X, y,model,criterion,args.epsilon,args.alpha)
            if args.attack == 'fgsm':
                Adv_x = half_attack_with_FGSM(X, y,model,criterion,args.epsilon,args.alpha)
                y_pred = model(Adv_x)
            elif args.attack == 'none':
                delta = torch.zeros_like(X) #加零等於沒攻擊
                Adv_x = X + delta
            elif args.attack == 'pgd':
                Adv_x = half_attack_with_PGD(X, y,model,criterion,opt,args.epsilon,args.attack_iters)
            
            '''
            if i == 0 :
                print("正常的")
                print(X)
                Adv_x = half_attack_with_PGD(X, y,model,criterion,opt,args.epsilon,args.attack_iters)
                print("被汙染的")
                print(Adv_x)
                
            else:
                break
            '''
            
           
            y_pred = model(Adv_x)
            y_true = y
            loss = criterion(y_pred, y_true)
            

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * y.size(0)
            train_n += y.size(0)

            y_pred , y_true = y_pred.cpu().detach().numpy() , y_true.cpu().detach().numpy()
            test_r2_score.append(r2_score(y_pred,y_true))

        avg_r2 = sum(test_r2_score) /len(test_r2_score)
        train_time = time.time()
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
            epoch, train_time - start_time, lr, train_loss/train_n, avg_r2)
        
        print(epoch, train_time - start_time, lr, train_loss/train_n, avg_r2)

        # torch.save(model.state_dict(), "pgd40_0.5_.pt")
        torch.save(model.state_dict(), args.modelPath)
    
def half_attack_with_FGSM(X, y,model,criterion,epsilon,alpha):
    rand_perm = np.random.permutation(X.size(0))
    rand_perm = rand_perm[:rand_perm.size//2]  #隨機取出一半的數值，放入攻擊內(加躁)。
    x_adv, y_adv = X[rand_perm,:], y[rand_perm]
    
    delta = FGSM_attack(x_adv,y_adv,model,criterion,epsilon,alpha)
    X[rand_perm,:] = x_adv + delta
    
    return X

def half_attack_with_PGD(X, y,model,criterion,opt,epsilon,attack_iters):
    rand_perm = np.random.permutation(X.size(0))
    rand_perm = rand_perm[:rand_perm.size]  #隨機取出一半的數值，放入攻擊內(加躁)。
    x_adv, y_adv = X[rand_perm,:], y[rand_perm]
    
    delta = PGD_attack(x_adv,y_adv,model,criterion,opt,epsilon,attack_iters)
    X[rand_perm,:] = x_adv + delta
    
    return X

if __name__ == "__main__":
    main()

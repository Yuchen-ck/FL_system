import argparse
import logging
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from uv_model import *
from preprocessing import *
from evaluate_attack  import *
from sklearn.metrics import r2_score


logger = logging.getLogger(__name__)
logging.basicConfig(
     filename = "eval_output.log",
    format='[%(asctime)s %(filename)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)

criterion = nn.MSELoss()




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default = 450, type=int)
    parser.add_argument('--attack', default='none', type=str, choices=['pgd', 'fgsm', 'none']) #pgd有問題
    parser.add_argument('--epsilon', default=0.5, type=float) #測試時的eps不能和train的eps差太多，若大於訓練值，準確度會大爆降
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=3, type=int)
    return parser.parse_args()


def main():
    print("---------------------------------------------")
    args = get_args()
    logger.info(args)

    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)


    Batch_size = args.batch_size
    test_loader = load_test_dataloader(Batch_size)

    
    model = Model().cuda()
    model.load_state_dict(torch.load('pgd_defense.pt'))
    # model.load_state_dict(torch.load('pgd_defense_0.3.pt'))
    # model.load_state_dict(torch.load('none_attack.pt')) 
    model.eval()

    total_loss = 0
    total_acc = 0
    n = 0
    
    test_r2_score = []
    if args.attack == 'none':
        with torch.no_grad():
            for i, (X, y,index) in enumerate(test_loader):
                X, y = X.cuda(), y.cuda()
                output = model(X)
                loss = criterion(output, y)
                total_loss += loss.item() * y.size(0)
                # total_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
                output, y =  output.cpu().detach().numpy() , y.cpu().detach().numpy()
                test_r2_score.append(r2_score(output, y))
            avg_r2 = sum(test_r2_score) /len(test_r2_score)
            print(avg_r2)
    else:
        for i,(X, y,index) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            
            if args.attack == 'pgd':
                Adv_x = half_attack_with_PGD(X, y,model,criterion,args.epsilon,args.alpha,args.attack_iters,args.restarts)
            
            elif args.attack == 'fgsm':
                Adv_x = half_attack_with_FGSM(X, y,model,criterion,args.epsilon)
            
            with torch.no_grad():
                output = model(Adv_x)
                loss = criterion(output, y)
                total_loss += loss.item() * y.size(0)
                # total_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
                output, y =  output.cpu().detach().numpy() , y.cpu().detach().numpy()
                test_r2_score.append(r2_score(output, y))
            
            avg_r2 = sum(test_r2_score) /len(test_r2_score)
            # print(avg_r2)

    logger.info('Test Loss: %.4f, Acc: %.4f', total_loss/n, avg_r2)

def half_attack_with_FGSM(X, y,model,criterion,epsilon): 
    rand_perm = np.random.permutation(X.size(0))
    rand_perm = rand_perm[:rand_perm.size//3]  #隨機取出一半的數值，放入攻擊內(加躁)。
    x_adv, y_adv = X[rand_perm,:], y[rand_perm]
    delta = attack_fgsm(x_adv, y_adv ,model , criterion , epsilon)
    X[rand_perm,:] = x_adv + delta
    return X

def half_attack_with_PGD(X, y,model,criterion,epsilon,alpha,attack_iters,restarts):
    rand_perm = np.random.permutation(X.size(0))
    # print(rand_perm.size//2)
    
    rand_perm = rand_perm[:rand_perm.size]  #隨機取出一半的數值，放入攻擊內(加躁)。
    x_adv, y_adv = X[rand_perm,:], y[rand_perm]
    
    delta = attack_pgd(x_adv, y_adv ,model ,criterion ,epsilon ,alpha ,attack_iters ,restarts)
    X[rand_perm,:] = x_adv + delta
    
    return X

if __name__ == "__main__":

    main()

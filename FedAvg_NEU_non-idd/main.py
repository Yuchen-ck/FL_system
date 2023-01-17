# -*- coding:utf-8 -*-
"""
@Time: 2022/02/14 12:11
@Author: KI
@File: fedavg-pytorch.py
@Motto: Hungry And Humble
"""
from uv_preprocessing_ import *
from uv_local_train import *
from uv_model import *
from fedAVG import *
from log_utils import *
# from uv_training import *
import torch
import time
import datetime

today = datetime.datetime.today()

def training_times(start,end):
    seconds = end-start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return (h, m, s)

def argparse_():
    import argparse

    #在這邊調參比較快
    dataset = "NEU"
    communication_round = 5
    total_clients = 10
    client_rate = 1
    non_iid = 1

    client_epoch = 10
    model = 'CNN'
    optimizer='Adam'
    lr = 0.0005
    
    #attack
    attack_type = "pgd"
    attack_rate = 0.3
    eps = 0.5    #pgd: 2/225開始
    
    attack_fixed = "non-fixed"

    parser = argparse.ArgumentParser(description='Adversarial Robusteness in Federated Learning Learning')
    parser.add_argument('-dataset', '--dataset',default = dataset, choices=['uv','NEU', 'mnist', 'china_steel'], help='adversarial attack type')

    parser.add_argument('-K', '--total_clients', default = total_clients, type=int, help='total client numbers')
    parser.add_argument('-cround', '--communicate_round', default = communication_round, type=int, help='communication Round')
    parser.add_argument('-E', '--client_epoch', default = client_epoch, type=int, help='client training epoch')
    parser.add_argument('-non_iid', '--non_iid', default = non_iid, type=int, help='0/1/2 , default is iid .')

    parser.add_argument('-rate', '--client_rate', default = client_rate, type=float,help='client joint aggregation rate')
    parser.add_argument('-m', '--model', default = model, choices=['CNN', 'ResNet18' ], help='model architecture')
    parser.add_argument('-opt', '--optimizer', default = optimizer, choices=['Adam', 'RMSprop'], help='optimizer select')
    parser.add_argument('-client_lr', '--lr', default = lr, type=float, help='client learning rate')
    
    parser.add_argument('-perturb_rate', '--perturb_rate', default = attack_rate, type=float,help='client perturbation rate')
    parser.add_argument('-eps', '--eps', default = eps, type=float,help='eplison')
    parser.add_argument('-attack', '--attack', default = attack_type, choices=['none','fgsm', 'pgd','others'], help='adversarial attack type')
    parser.add_argument('-attack_random_type', '--attack_fixed', default = attack_fixed, choices=['fixed', 'non-fixed'], help='adversarial attack type') 
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argparse_()

    start = time.time() #開始計時
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fed = FedAvg(clients_num = args.total_clients ,device = device,client_epoch = args.client_epoch, non_idd = args.non_iid , model = args.model ,optimizer = args.optimizer ,lr=args.lr ,perturb_rate = args.perturb_rate,attack = args.attack ,attack_fixed = args.attack_fixed ,eps = args.eps)
    
    fed.server(r = args.communicate_round , client_rate = args.client_rate, dataset = args.dataset)
    
    model,test_r2_score = fed.server_test()

    model_name = str(today.strftime('%Y-%m-%d')) + f"_{test_r2_score}.pth"
    torch.save(model.state_dict(), model_name )

    end = time.time()  #結束計時

    train_times = training_times(start,end)   #印出訓練總長度 
    
    #寫入log
    client_number = args.total_clients
    client_rate = args.client_rate
    client_epoch = args.client_epoch
    r = args.communicate_round 
    non_idd = args.non_iid
    attack  =args.attack
    dataset = args.dataset
    
    save_result(dataset,train_times,test_r2_score ,client_number ,client_rate ,client_epoch , r ,non_idd ,attack)
    
    # clients_numbers = 10
    # fed.global_test(clients_numbers)
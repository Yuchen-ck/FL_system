# -*- coding:utf-8 -*-
"""
@Time: 2022/02/14 12:11
@Author: KI
@File: fedavg-pytorch.py
@Motto: Hungry And Humble
"""
from preprocessing import *
from train_test_utils import *
from model import *
from fedAVG import *
from log_utils import *
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
    dataset = "UV_regression"

    communication_round = 30
    total_clients = 10
    client_rate = 0.8
    non_iid = 1

    client_epoch = 10
    model = 'ANN'
    optimizer='RMSprop'
    lr = 0.1
    
    #attack
    attack_type = "fgsm"
    eps = 0.3     #pgd: 從2/225開始
    perturb_rate = 0.1
    attack_fixed = "non-fixed"

    parser = argparse.ArgumentParser(description='Adversarial Robusteness in Federated Learning Learning')

    parser.add_argument('-dataset', '--dataset', default = dataset, choices=['mnist', 'UV_regression'], help='dataset select')
    parser.add_argument('-K', '--total_clients', default = total_clients, type=int, help='total client numbers')
    parser.add_argument('-cround', '--communicate_round', default = communication_round, type=int, help='communication Round')
    parser.add_argument('-E', '--client_epoch', default = client_epoch, type=int, help='client training epoch')
    parser.add_argument('-non_iid', '--non_iid', default = non_iid, type=int, help='0/1/2 , default is iid .')

    parser.add_argument('-rate', '--client_rate', default = client_rate, type=float,help='client joint aggregation rate')
    parser.add_argument('-m', '--model', default = model, choices=['ANN', 'small_ANN'], help='model architecture')
    parser.add_argument('-opt', '--optimizer', default = optimizer, choices=['adam', 'RMSprop'], help='optimizer select')
    parser.add_argument('-client_lr', '--lr', default = lr, type=float, help='client learning rate')
    
    parser.add_argument('-perturb_rate', '--perturb_rate', default = perturb_rate, type=float,help='client perturbation rate')
    parser.add_argument('-eps', '--eps', default = eps, type=float,help='eplison')
    parser.add_argument('-attack', '--attack_type', default = attack_type, choices=['none','fgsm', 'pgd','others'], help='adversarial attack type')
    parser.add_argument('-attack_random_type', '--attack_fixed', default = attack_fixed, choices=['fixed', 'non-fixed'], help='adversarial attack type') 
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argparse_()

    start = time.time() #開始計時
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fed = FedAvg(clients_num = args.total_clients ,device = device,client_epoch = args.client_epoch, non_idd = args.non_iid , model = args.model ,optimizer = args.optimizer ,lr=args.lr ,perturb_rate = args.perturb_rate,attack = args.attack_type ,attack_fixed = args.attack_fixed ,eps = args.eps)
    
    fed.server(r = args.communicate_round , client_rate = args.client_rate , dataset = args.dataset)
    
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
    attack  =args.attack_type
    
    save_result(train_times,test_r2_score ,client_number ,client_rate ,client_epoch , r ,non_idd ,attack)
    
    # clients_numbers = 10
    # fed.global_test(clients_numbers)
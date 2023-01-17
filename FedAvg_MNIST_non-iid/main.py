# -*- coding:utf-8 -*-
"""
@Time: 2022/02/14 12:11
@Author: KI
@File: fedavg-pytorch.py
@Motto: Hungry And Humble
"""
from preprocessing import *
from model import *
from fedAVG import *

import time
# 加上計算訓練時間


if __name__ == '__main__':
    start = time.time() #開始計時

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fed = FedAvg(clients_num =10 ,device = device)
    fed.server(r = 100 , client_rate = 0.8)
    
    model = fed.server_test()

    print(model)

    end = time.time() #結束計時

    seconds = end-start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("Running time: %d:%02d:%02d" % (h, m, s))
    
    # clients_numbers = 10
    # fed.global_test(clients_numbers)
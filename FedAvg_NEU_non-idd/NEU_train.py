import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from PIL import Image
import torch 
from torch import nn
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from NEU_preprocessing import *

import copy
def train_NEU(model,non_iid,lr,optimizer_name,client_epoches,trained_clinet_number,total_clients):
    device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
    model.to(device)
    model.train() #開啟訓練模式
    train_bz=4
    train_loader  = load_train_dataloader(trained_clinet_number ,total_clients,non_iid,train_bz)
    test_loader = load_test_datalaoder(val_bz=1)
    model.len = len(train_loader)
    

    criterion = nn.CrossEntropyLoss()

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
                              
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_losses = []
    test_correct = []
    
    for i in range(client_epoches):
        trn_corr = 0
        tst_corr = 0
        acc = 0
        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):
            b+=1
            (X_train, y_train) = (X_train.to(device), y_train.to(device))
            # Apply the model
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)

            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            # accuracy = predicted.eq(y_train.data).sum()/ train_bz*100 
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print interim results
            num = 100
            if b%num== 0:
                print(f'epoch: {i:2}  batch: {b:4} [{b*train_bz:6}/{len(train_loader)}]  loss: {loss.item():10.8f}')

        train_losses.append(loss.item())
        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, gt) in enumerate(test_loader):
                # Limit the number of batches
                (X_test, gt) = (X_test.to('cuda'), gt.to('cuda'))
                # Apply the model
                y_val = model(X_test)

                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1] 
                
                # 設定batch size = 1 的時候，才能使用!!!
                if predicted[0] ==  gt.data[0] :
                    acc += 1
            print(acc)       

        # if i % 10 == 0 :
        #     torch.save(net.state_dict(), f'PNA{i}.pt')
        
        acc_2 = acc / 360
        test_correct.append(acc_2)
        print(acc_2)
        scheduler.step()
        #深複製模型
        model.train()
        best_model = copy.deepcopy(model)
    
    return best_model
        

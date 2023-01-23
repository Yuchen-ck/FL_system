import argparse
import logging
import time
from itertools import chain


#from args import args_parser
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision.models as models

import pandas as pd
import os


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score

#自定義創立資料集的方法
class DataMaker(Dataset):
    def __init__(self, X, y):
        # scaler = StandardScaler()
        scaler = preprocessing.StandardScaler()
        self.targets = scaler.fit_transform(X.astype(np.float32))
        self.labels = y.astype(np.float32)

    
    def __getitem__(self, i): 
        # i不一定要使用到，這裡(魔術方法)也會被調用
        # 在這裡換成tensor(?)
        self.targets_ = torch.from_numpy(self.targets)
        self.labels_ = torch.tensor(self.labels.values)
        
        # print("---------魔術方法 go ahead----------------")
        # print(self.targets_)
        # print(type(self.labels))
        # print(len(self.targets_))
        # print("---------魔術方法 over----------------")

        return self.targets_, self.labels_
        
    def __len__(self):
        return len(self.targets)

# 資料讀取的處理
def load_data(file_name): 
    #df = pd.read_csv(os.path.dirname(os.getcwd()) + "\\"+file_name , encoding='gbk')
    df = pd.read_csv(file_name , encoding='gbk') 
    df= df.reset_index(drop = True)
    df = df.drop(df.columns[[0]], axis = 1)
    return df

#把資料放進DataLoader
def data_preprocessing(file_name, Batch_size): #nn_seq_wind: 本來的名稱
    print('data processing...')
    df_train = load_data(file_name)

    train_split_rate = 0.85

    x = df_train.iloc[:,0:3]
    y = df_train.iloc[:,-1:] 
    X_train,X_test, y_train, y_test = train_test_split(x,y ,test_size= (1-train_split_rate))

    
    train_set = DataMaker(X_train, y_train)
    test_set = DataMaker(X_test, y_test)

    # print(len(train_set)) #印出訓練集數量
    # print(len(test_set))  #印出測試集數量
    # print(train_set['training data']) #魔術方法__getitem__的使用方法
    # # print(test_set['testing data']) #魔術方法__getitem__的使用方法


    #打包成dataloader
    train_loader = DataLoader(train_set, batch_size = Batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size = Batch_size, shuffle=False) 

    n_features = x.shape[1]

    return train_loader,  test_loader , n_features

class Model(nn.Module):
    def __init__(self, n_features):
        super(Model, self).__init__()
        self.linearA = nn.Linear(n_features, 20)
        self.linearB = nn.Linear(20, 16)
        self.linearC = nn.Linear(16, 4)
        self.linearD = nn.Linear(4, 1)

    def forward(self, x):
        X = F.relu(self.linearA(x))
        X = F.relu(self.linearB(X))
        X= F.relu(self.linearC(X))
        return self.linearD(X)

def apply_model(n_features):
    FCLmodel = Model(n_features)
    criterion = nn.L1Loss() #mean absolute error
    #optimizer = torch.optim.SGD(FCLmodel.parameters(), lr=0.001 ,momentum=0.9, decay= 1e-6)
    optimizer = torch.optim.Adam(FCLmodel.parameters(), lr=0.01)
    return FCLmodel ,criterion ,optimizer

#計算模型參數
def count_parameters(model): 
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>8}')
    print(f'________\n{sum(params):>8}')


def train(training_loader,n_epochs,criterion ,optimizer,n_features,FCLmodel):
    print("進入train process")
    train_loader = training_loader
    epochs = n_epochs

    max_trn_batch = 850
    train_losses =[]
    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0
         # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):
            b+=1

            #Attack

            # Apply the model
            y_pred = FCLmodel(X_train)
            loss = criterion(y_pred, y_train)

            
            # # Tally the number of correct predictions
            # predicted = torch.max(y_pred.data, 1)[1]
            # batch_corr = (predicted == y_train).sum()
            # trn_corr += batch_corr

            #  # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b%10 == 0:
                print(f'epoch: {i:2}  batch: {b:4} [{50*b:6}/8000]  loss: {loss.item():10.8f} ')

        train_losses.append(loss.item())
        torch.save(FCLmodel, './regression_model.pth') #保存完整模型
        # torch.save(FCLmodel.state_dict(), './regression_model.pth') #只保存模型權重
        
    return train_losses


def test(testing_loader,FCLmodel):
    test_loader = testing_loader

    with torch.no_grad(): #這行之後才開始測試
        for b, (X_test, y_test) in enumerate(test_loader):
            # Limit the number of batches
            if b == 850:
                break

            # Apply the model
            y_val = FCLmodel(X_test)
            
    return y_val , y_test

def load_model():
    # net = models.squeezenet1_1(pretrained=False)
    pthfile = './regression_model.pth'
    net = torch.load(pthfile)
    print(net)
    return net


#印出dataloader內的東西
def print_dataloader(training_loader):
    data_loader = training_loader

    for images,lables in data_loader:
        break
    print(images)
    print(lables)
        

def plt_loss_graph(train_losses):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(train_losses, label='training loss')
    plt.title('Loss at the end of each epoch')
    plt.legend()
    fig.savefig('./plot.png')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', default='none', type=str, choices=['none', 'pgd', 'fgsm'])
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    #打包成dataloader的訓練集 #打包成dataloader的測試集 #n_features: for 訓練模型架構
    training_loader , testing_loader , n_features = data_preprocessing("dataset.csv", Batch_size = 50)

    #印出dataloader內的資料
    #print_dataloader(training_loader)
    
    #建立訓練模型
    net = Model(n_features)
    #計算模型參數
    count_parameters(net) 
    #
    FCLmodel ,criterion ,optimizer = apply_model(n_features)
    print(FCLmodel)


    #開始訓練
    n_epochs = 20
    train_losses = train(training_loader ,n_epochs ,criterion ,optimizer ,n_features,FCLmodel )
    plt_loss_graph(train_losses)
    
    # net = load_model()
    #開始測試r2_score效果
    y_val , y_test = test(testing_loader,FCLmodel)
    # y_val , y_test = test(testing_loader,net)
    y_val = y_val.view(50,-1)
    y_test = y_test.view(50,-1)
   
    r2_score_list =[]
    for i in range(y_test.shape[0]):
        r2_score_list.append(r2_score(y_val[i],y_test[i]))
    print("testing R2_score: ")
    print(sum(r2_score_list) /len(r2_score_list))
    print("ok")
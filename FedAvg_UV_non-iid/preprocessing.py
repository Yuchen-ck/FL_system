from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# 创建MyDataset类
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.data = torch.from_numpy(x).float()
        self.label = torch.from_numpy(y).float()
 
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], idx
 
    def __len__(self):
        return len(self.data)

def convert_to_dataloader(file_name):
    df = pd.read_csv(file_name)
    # features are in cols [0,3], median price in [4]
    x = df.iloc[:,0:2]
    y = df.iloc[:,-1:]
    
    #dataframe to np.array
    Train_data = x.values
    Train_label = y.values

    # 歸一化: [ 0，1 ] 
    min_max_scaler = preprocessing.MinMaxScaler()
    Train_data = min_max_scaler.fit_transform(Train_data)
    
    data_set = MyDataset(Train_data, Train_label)
    print('len:', len(data_set))

    return data_set
    
def convert_to_dataloader_non_iid(file_name,split_num):
    df = pd.read_csv(file_name)
    # features are in cols [0,2], median price in [4]

    df = df.head(int(split_num))
    x = df.iloc[:,0:2]
    y = df.iloc[:,-1:]
    
    #dataframe to np.array
    Train_data = x.values
    Train_label = y.values

    # 歸一化: [ 0，1 ] 
    min_max_scaler = preprocessing.MinMaxScaler()
    Train_data = min_max_scaler.fit_transform(Train_data)
    
    data_set = MyDataset(Train_data, Train_label)
    print('len:', len(data_set))

    return data_set

def convert_to_dataloader_non_iid_v2(file_name,avg_len,user_id):
    df = pd.read_csv(file_name)
    # features are in cols [0,2], median price in [4]
    

    df = df.iloc[int(user_id*avg_len) : int((user_id+1)*avg_len)]
    x = df.iloc[:,0:2]
    y = df.iloc[:,-1:]
    
    #dataframe to np.array
    Train_data = x.values
    Train_label = y.values

    # 歸一化: [ 0，1 ] 
    min_max_scaler = preprocessing.MinMaxScaler()
    Train_data = min_max_scaler.fit_transform(Train_data)
    
    data_set = MyDataset(Train_data, Train_label)
    print('len:', len(data_set))

    return data_set

def load_train_dataloader(trained_clinet_number ,total_clients,Batch_size,non_iid):
    
    user_id = trained_clinet_number
    users_count = total_clients
    batch_sz= Batch_size


    if non_iid == 0: #每個client的資料量相同，但資料是隨機的
        file_name = "C:/Users/user/Documents/GitHub/FL_implement/dataset/train_set.csv"
        train_set = convert_to_dataloader(file_name)

        sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=users_count, rank=user_id)
        train_loader = torch.utils.data.DataLoader(train_set, sampler=sampler,batch_size= batch_sz, shuffle=sampler is None)
    
    elif non_iid == 1:#non-iid: client資料分布不夠廣
        print(user_id)
        file_name = "C:/Users/user/Documents/GitHub/FL_implement/dataset/train_set.csv"
        train_len = 16000
        avg_len =   train_len / users_count
        train_set = convert_to_dataloader_non_iid(file_name, avg_len*(user_id+1))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_sz, shuffle=True)

       
    else: 
        print("mode 2: ", user_id)
        file_name = "C:/Users/user/Documents/GitHub/FL_implement/dataset/train_set.csv"
        train_len = 16000
        avg_len =   train_len / users_count
        train_set = convert_to_dataloader_non_iid_v2(file_name, avg_len ,user_id)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_sz, shuffle=True)
        

    return train_loader



def load_test_dataloader(Batch_size):
    #1221: 讀檔名改成用for loop去讀(?
    file_name = "C:/Users/user/Documents/GitHub/FL_implement/dataset/test_set.csv"
    test_set = convert_to_dataloader(file_name)

    batch_sz= Batch_size

    test_loader = DataLoader(test_set, batch_size = batch_sz, shuffle=False, num_workers=2) 

    return test_loader
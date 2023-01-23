import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
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
    # df = df.head(50)
    # features are in cols [0,3], median price in [4]
    x = df.iloc[:,0:3]
    y = df.iloc[:,-1:]
    

    #dataframe to np.array
    Train_data = x.values
    Train_label = y.values

    # # 歸一化: [ 0，1 ] 
    min_max_scaler = preprocessing.MinMaxScaler()
    Train_data = min_max_scaler.fit_transform(Train_data)
    
    data_set = MyDataset(Train_data, Train_label)
    print('len:', len(data_set))

    return data_set
    


def load_train_dataloader(batch_sz):
    file_name = "all_val.csv"
    train_set = convert_to_dataloader(file_name)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_sz, shuffle=True)

    return train_loader

def load_test_dataloader(Batch_size):
    
    file_name = "all_test.csv"
    test_set = convert_to_dataloader(file_name)
    batch_sz= Batch_size
    test_loader = DataLoader(test_set, batch_size = batch_sz, shuffle=False, num_workers=2) 

    return test_loader



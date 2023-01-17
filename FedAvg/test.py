from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math

def combine_df(n1,n2,n3):
    file1 = f"./uv_client/client_{n1}.csv"
    file2 = f"./uv_client/client_{n2}.csv"
    file3 = f"./uv_client/client_{n3}.csv"
    df1 = pd.read_csv(file1 , encoding='gbk')
    df2 = pd.read_csv(file2 , encoding='gbk')
    df3 = pd.read_csv(file3 , encoding='gbk')
    # print(df1.head(5))
    # print(df2.head(5))
    # print(df3.head(5))
    
    dfs = [df1, df2, df3]
    combine_df = pd.concat(dfs)
    filename = f"./uv_client_v2/client_{n2}.csv"
    combine_df.to_csv(filename)  


if __name__ == '__main__':

    for user_id in range(0,10):
        if user_id == 0:
            combine_df(9,0,1)
        elif  user_id == 9:
            combine_df(8,9,0)
        else:
            combine_df(user_id-1,user_id,user_id+1)



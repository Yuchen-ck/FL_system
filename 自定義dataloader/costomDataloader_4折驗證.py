import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets


class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, indices,transforms=None):
        # __init__() function is where the initial logic happens like reading a csv, assigning transforms, filtering data, etc.
        # 1. read csv file 
        # 第13~16行 csv file的前處理
        df = pd.read_csv(csv_path,usecols=["image_name","target"])
        # df = df.head(5)
        first_column = df.iloc[:, 0].apply(lambda x: x + ".jpg")  #改成.jpg結尾
        last_column = df.iloc[:, -1]
        df = pd.concat([first_column, last_column], axis=1)
        self.indices = indices

        new_df = df.iloc[self.indices.tolist()]
        print(new_df)

        self.data_info = new_df.copy(deep=False)

        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.labels_arr = np.asarray(self.data_info.iloc[:, 1])

        # Calculate len
        self.data_len = len(self.data_info.index)

        # print(self.image_arr.shape)
        # print(self.labels_arr)
        
        self.transforms = transforms #不要在__init__裡面做transforms，不然塞太多東西在這裡。


    def __getitem__(self, index):
        # __getitem__() function returns the data and labels. This function is called from dataloader like this:
        
        # Get image name from the pandas df
        single_image_path = "./medical/" + self.image_arr[index]
        # Open image
        img_as_numpy = Image.open(single_image_path)
        img_as_tensor  = self.transforms(img_as_numpy)
        
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels_arr[index]
        label_as_tensor = torch.tensor(single_image_label)
        # print(type(label_as_tensor))
        return (img_as_tensor, label_as_tensor)

    def __len__(self):
        # return the length of dataset #資料集總長度
        dataset_len = self.data_len
        return dataset_len 

if __name__=='__main__':
    
    # Define transforms 
    height = width = 32
    to_tensor = transforms.ToTensor()
    resize_tensor = transforms.Resize((height, width))
    normalize_tensor = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                     std=[1/0.229, 1/0.224, 1/0.225])
    transformations = transforms.Compose([to_tensor,resize_tensor,normalize_tensor])
    
    # Define custom dataset
    batch_size = 64
    fileTrainPath =  "./train.csv"
    df = pd.read_csv(fileTrainPath,usecols=["image_name","target"])
    total_len = len(df)


    # custom_dataset_from_csv = CustomDatasetFromCSV(fileTrainPath, transformations)
    # print(custom_dataset_from_csv.__len__())
    # total_len = custom_dataset_from_csv.__len__()


    # 定義 K-fold 交叉驗證的 K 值
    K = 4

    # 將數據分成 K 個部分
    # num_samples = total_len
    num_samples = 20
    indices = np.arange(num_samples)
    splits = np.array_split(indices, K)

    for k in range(K):
        # 取出驗證集和訓練集
        validation_indices = splits[k]
        train_indices = np.concatenate([splits[i] for i in range(K) if i != k])

        customTrainDataset = CustomDatasetFromCSV(fileTrainPath ,train_indices ,transformations)
        print("訓練總長度:")
        print(customTrainDataset.__len__())

        customValDataset = CustomDatasetFromCSV(fileTrainPath ,validation_indices ,transformations)
        print("驗證總長度:")
        print(customValDataset.__len__())
        



        



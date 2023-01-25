import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data import DataLoader
import random,numpy
import pandas as pd

# load image from csv file
class CustomDatasetFromCSV(Dataset):
    def __init__(self, df, transforms=None):
        # __init__() function is where the initial logic happens like reading a csv, assigning transforms, filtering data, etc.
        # 1. use df 
        # first_column = df.iloc[:, 0].apply(lambda x: x + ".jpg")  #改成.jpg結尾
        first_column = df.iloc[:, 0]
        last_column = df.iloc[:, -1]
        df = pd.concat([first_column, last_column], axis=1)
        
        self.data_info = df.copy(deep=False)

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
        # single_image_path = "C:/Users/user/Documents/GitHub/FL_implement/NEU-CLS/dataset/all_image/" + self.image_arr[index]
        single_image_path = "./image_dataset/" + self.image_arr[index]
        # Open image
        image = Image.open(single_image_path)
        image = image.convert("RGB")  #單通道圖片轉成3通道圖片
        img_as_numpy = np.array(image)

        img_as_tensor  = self.transforms(img_as_numpy)
        # print(img_as_tensor.shape)
        
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels_arr[index]
        label_as_tensor = torch.tensor(single_image_label)
        # print((label_as_tensor))
        return (img_as_tensor, label_as_tensor)

    def __len__(self):
        # return the length of dataset #資料集總長度
        dataset_len = self.data_len
        return dataset_len 

# transform  
class RandRotateTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)
        
def transformation(): # 圖片大小 # 100*100
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((100,100)),
                            RandRotateTransform(angles=[0, 90, 180, 270]),
                            transforms.RandomHorizontalFlip(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
    ])

    return transform

def read_and_split_df(csv_path):
    df = pd.read_csv(csv_path,usecols=["image_name","target"])
    df = df.head(5)
    return df

if __name__ == '__main__':
    transform = transformation()
    trainPath = "./csv_dataset/train.csv"
    train_df = read_and_split_df(trainPath)
    customTrainDataset = CustomDatasetFromCSV(train_df,transform)
    
    sampler = torch.utils.data.distributed.DistributedSampler(customTrainDataset, num_replicas=10, rank=1)
    train_loader = torch.utils.data.DataLoader(customTrainDataset, sampler=sampler,batch_size= 4, shuffle=sampler is None)
    
    for b, (X_test, gt) in enumerate(train_loader):
        print(X_test , gt)
        break
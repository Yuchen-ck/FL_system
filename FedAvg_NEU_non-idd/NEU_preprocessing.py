import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data import DataLoader
import random,numpy

# load image from csv file
class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, indices,transforms=None):
        # __init__() function is where the initial logic happens like reading a csv, assigning transforms, filtering data, etc.
        # 1. read csv file 
        # 第13~16行 csv file的前處理
        df = pd.read_csv(csv_path,usecols=["image_name","target"])
        # df = df.head(5)
        # first_column = df.iloc[:, 0].apply(lambda x: x + ".jpg")  #改成.jpg結尾
        first_column = df.iloc[:, 0]
        last_column = df.iloc[:, -1]
        df = pd.concat([first_column, last_column], axis=1)
        self.indices = indices
        
        new_df = df.iloc[self.indices.tolist()]
        # print(new_df)

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
        single_image_path = "C:/Users/user/Documents/GitHub/FL_implement/NEU-CLS/dataset/all_image/" + self.image_arr[index]
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

# split train and test dataset
def random_list():
    import random
    import numpy 
    import copy

    val_numbers = []
    random.seed(69) #每次隨機都一樣
    for i in range(6):
        numbers = range(300*(i) ,300*(i+1))
        selected_numbers = sorted(random.sample(numbers,60)) #從每類中隨機挑出45張圖片，當成測試集(60*6=360)
        # remaining_numbers = sorted(set(numbers) - set(selected_numbers))
        # selected_numbers += selected_numbers
        # print(selected_numbers)
        val_numbers.append(selected_numbers)

    from itertools import chain
    flattened_list = list(chain(*val_numbers))
    # print(flattened_list)
    # print(len(flattened_list))

    numbers = range(0 ,1800)
    remaining_numbers = sorted(set(numbers) - set(flattened_list))

    train_nubmers = copy.copy(remaining_numbers)
    val_numbers = copy.copy(flattened_list)

    train_nubmers = numpy.array(train_nubmers)
    val_numbers = numpy.array(val_numbers)
    
    return train_nubmers , val_numbers


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


def convert_to_custom_Dataset(list_indices):
    transform = transformation()
    filePath = "C:/Users/user/Documents/GitHub/FL_implement/NEU-CLS/dataset/all_image/dct.csv"
    customDataset = CustomDatasetFromCSV(filePath ,list_indices ,transform)

    return customDataset

def non_iid_v1(train_indices_numpy,users_count,user_id):
    train_indices = train_indices_numpy.tolist()
    random.shuffle(train_indices)
    split_number = len(train_indices)//users_count
    split_indices = train_indices[0:split_number*(user_id+1)]
    split_indices_numpy = numpy.array(split_indices)

    print(f'{user_id}: 資料量 {len(split_indices)}')

    return split_indices_numpy
    

def load_train_dataloader(trained_clinet_number ,total_clients,non_iid,train_bz=4):
    # customTrainDataset , _  ,train_indices = convert_to_custom_Dataset()
    user_id = trained_clinet_number
    users_count = total_clients

    train_indices_numpy , _ =  random_list()

    if non_iid == 0: #每個client的資料量相同，但資料是隨機的
        customTrainDataset = convert_to_custom_Dataset(train_indices_numpy)
        sampler = torch.utils.data.distributed.DistributedSampler(customTrainDataset, num_replicas=users_count, rank=user_id)
        train_loader = torch.utils.data.DataLoader(customTrainDataset, sampler=sampler,batch_size= train_bz, shuffle=sampler is None)
    else: 
        split_indices_numpy = non_iid_v1(train_indices_numpy,users_count,user_id)
        
        custom_non_iid_Dataset = convert_to_custom_Dataset(split_indices_numpy)
        train_loader = DataLoader(custom_non_iid_Dataset, batch_size=train_bz, shuffle=True)
    
        

    # train_loader = DataLoader(customTrainDataset, batch_size=train_bz, shuffle=True)
    
    
   
    return train_loader 

def load_test_datalaoder(val_bz=1):

    _ ,test_indices =  random_list()
    customValDataset = convert_to_custom_Dataset(test_indices)
    
    test_loader = DataLoader(customValDataset, batch_size=val_bz, shuffle=True)

    return test_loader


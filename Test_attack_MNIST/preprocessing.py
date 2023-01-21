from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_MNIST_training_set(batch_size,trained_clinet_number,total_clients):
    bs = batch_size
    users_count = total_clients
    user_id = trained_clinet_number

    # data preprocessing
    data_tf = transforms.Compose(
        [transforms.ToTensor()]
        )
    

    # 使用内置函数下载mnist数据集
    train_set = datasets.MNIST('./MNIST_data', train=True, transform=data_tf, download=True)
    
    sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=users_count, rank=user_id)
    train_loader = torch.utils.data.DataLoader(train_set, sampler=sampler,batch_size=batch_size, shuffle=sampler is None)

    return train_loader



def load_MNIST_testing_set(batch_size):
    bs = batch_size

    # data preprocessing
    data_tf = transforms.Compose(
        [transforms.ToTensor()]
        )
    

    # 使用内置函数下载mnist数据集
    test_set = datasets.MNIST('./MNIST_data', train=False, transform=data_tf, download=True)
    
    len_of_test_dataset = len(test_set.targets) #gt #印出testing set長度

  
    test_loader = DataLoader(test_set, batch_size = bs, shuffle = False)

    return test_loader ,len_of_test_dataset
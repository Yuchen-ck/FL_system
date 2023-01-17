import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

bs = 1024
users_count = 10 
user_id = 1 

# data preprocessing
data_tf = transforms.Compose(
    [transforms.ToTensor()]
    )


# 使用内置函数下载mnist数据集
train_set = datasets.MNIST('./MNIST_data', train=True, transform=data_tf, download=True)
test_set = datasets.MNIST('./MNIST_data', train=False, transform=data_tf, download=True)

# print(len(train_set.data)) #images #印出training set長度
# print(len(test_set.targets)) #gt #印出testing set長度

sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=users_count, rank=1)
print(len(sampler))
train_loader_1 = torch.utils.data.DataLoader(train_set, sampler=sampler,batch_size= bs, shuffle=sampler is None)

sampler_2 = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=users_count, rank=2)
train_loader_2 = torch.utils.data.DataLoader(train_set, sampler=sampler_2,batch_size= bs, shuffle=sampler_2 is None)

if train_loader_1 == train_loader_2:
    print("same")
else:
    print("different")
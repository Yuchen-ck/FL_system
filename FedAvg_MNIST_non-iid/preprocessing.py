from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


'''
#自定義創立資料集的方法
class DataMaker(Dataset):
    def __init__(self, X, y):
        # scaler = StandardScaler()
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)) #正規化到0~1 #ReLu把負值拿掉，不需要1~-1
        self.targets = scaler.fit_transform(X.astype(np.float32))
        self.labels = y.astype(np.float32)
    
    def __getitem__(self, i):
        self.targets_ = torch.from_numpy(self.targets)
        self.labels_ = torch.tensor(self.labels.values)
        return self.targets_, self.labels_

    def __len__(self):
        return len(self.targets)

# 資料讀取的處理
def load_data(file_name): 
    #df = pd.read_csv(os.path.dirname(os.getcwd()) + "\\"+file_name , encoding='gbk')
    df = pd.read_csv("../"+file_name , encoding='gbk')
    df= df.reset_index(drop = True)
    df = df.drop(df.columns[[0]], axis = 1)
    return df

#把資料放進DataLoader
def data_preprocessing(file_name, Batch_size): #nn_seq_wind: 本來的名稱
    #print('data processing...')
    df_train = load_data(file_name)

    train_split_rate = 0.85

    x = df_train.iloc[:,0:3]
    y = df_train.iloc[:,-1:] 
    X_train,X_test, y_train, y_test = train_test_split(x,y ,test_size= (1-train_split_rate))
    X_train,X_val, y_train, y_val = train_test_split(X_train,y_train ,test_size= (1-train_split_rate))


    train_set = DataMaker(X_train, y_train)
    test_set = DataMaker(X_test, y_test)
    val_set = DataMaker(X_val, y_val)

    train_loader = DataLoader(train_set, batch_size = Batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size = Batch_size, shuffle=True) 
    val_loader = DataLoader(test_set, batch_size = Batch_size, shuffle=True)

    n_features = x.shape[1]

    return train_loader,  test_loader , val_loader , n_features
'''

def load_MNIST_training_set(batch_size,trained_clinet_number,total_clients,non_iid):
    bs = batch_size
    users_count = total_clients
    user_id = trained_clinet_number

    # data preprocessing
    data_tf = transforms.Compose(
        [transforms.ToTensor()]
        )

    train_set = datasets.MNIST('./MNIST_data', train=True, transform=data_tf, download=True)
    
    if non_iid == 0:
        sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=users_count, rank=user_id)
        train_loader = torch.utils.data.DataLoader(train_set, sampler=sampler,batch_size=batch_size, shuffle=sampler is None)
    
    elif non_iid == 1:
        if user_id ==0 :
            id_dataset = [(x, y) for (x, y) in train_set if y == user_id+9 or y == user_id or y == user_id+1] # 9-0-1
        elif user_id == 9 :
            id_dataset = [(x, y) for (x, y) in train_set if y == user_id-1 or y == user_id or y == user_id-9] # 8-9-0
        else:
            id_dataset = [(x, y) for (x, y) in train_set if y == user_id-1 or y == user_id or y == user_id +1]
        print(len(id_dataset))
        train_loader = torch.utils.data.DataLoader(id_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    else: #一個client只包含一種類別
        id_dataset = [(x, y) for (x, y) in train_set if y == user_id]
        print(len(id_dataset))
        train_loader = torch.utils.data.DataLoader(id_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    

    return train_loader



def load_MNIST_testing_set(batch_size):
    bs = batch_size

    # data preprocessing
    data_tf = transforms.Compose(
        [transforms.ToTensor()]
        )
    

    # 使用内置函数下载mnist数据集
    test_set = datasets.MNIST('./MNIST_data', train=False, transform=data_tf, download=True)
    
    print(len(test_set.targets)) #gt #印出testing set長度

  
    test_loader = DataLoader(test_set, batch_size = bs, shuffle = False)

    return test_loader
from model import *
from train_test_utils import * 
import copy
import random
import torch 
import numpy as np


class FedAvg:
    def __init__(self, clients_num ,device): # def __init__(self, args):
        self.client_number = clients_num
        self.device = device
        self.clients_list = []  #10 clients
        for i in range(self.client_number):
            self.clients_list.append("client_"+str(i))
        
        #Load MNIST Model
        self.nn =  CNN_Model() 
        self.nns = []
        

        for i in range(self.client_number):
            temp = copy.deepcopy(self.nn)
            #temp.name = self.clients_list[i] #不知道這是啥小
            self.nns.append(temp)

        

    def server(self, r, client_rate):
        self.r = r  #the number of communication rounds
        self.client_rate = client_rate
        self.non_iid = 2 # 資料是否為非獨立分布，1代表non-iid 
        for t in range(self.r):
            print('round', t + 1, ':') #最外圈for loop
            # sampling
            m = np.max([int(self.client_rate * self.client_number), 1])
            index = random.sample(range(0, self.client_number), m)  #St #random set of m clients
            
            # 1. dispatch
            self.dispatch(index)

            # 2. local updating
            self.client_update(index,self.device,self.non_iid)
           
            # 3. aggregation
            self.aggregation(index)

        return self.nn


    #1.
    def dispatch(self, index):
        print(index) 

        #print(self.nns[0].len) #從這裡改

        for j in index:
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone()
                #print(old_params.data)

    #訓練client端模型
    #2. 
    def client_update(self, index, device ,non_iid):  # update nn
        # print("開始更新個別client")
        self.device = device
        total_clients = self.client_number

        if non_iid == 0:
            print("client資料為iid，簡單分配。")
        else:
            print("client資料為non-iid")
        
        E = 5
        optimizer= 'adam'
        for k in index:
            print("The client_{} start to train".format(k))
            self.nns[k] = train(self.nns[k],non_iid,optimizer_name= optimizer,client_epoches = E, trained_clinet_number=k ,total_clients= total_clients)
        
        
    #3.
    def aggregation(self, index): #question
        s = 0
        for j in index:
            # normal
            s += self.nns[j].len #s: 總資料量

        params = {}
        for name,  param in self.nns[0].named_parameters(): # model的名字與参数迭代器
            params[name] = torch.zeros_like(param.data)   

        for j in index:
            for name, param in self.nns[j].named_parameters(): 
                params[name] += param.data * (self.nns[j].len / s)

        for k, v in self.nn.named_parameters():
            v.data = params[k].data.clone()
            

    def server_test(self):
        model = self.nn
        server_model = copy.deepcopy(model)
        server_model.eval()

        server_name = ["server"]
        
        for server in server_name:
            server_model.name = server
            print("server測試結果為: ")
            test(model)
        
        torch.save(model.state_dict(),"non-iid_v2.pth")
    
        return model
        


    def global_test(self,clients_num):
        model = self.nn
        model.eval()
        clients_wind = ['Task1_W_Zone' + str(i) for i in range(1, clients_num)]
        c = clients_wind
        for client in c:
            model.name = client
            test(model)
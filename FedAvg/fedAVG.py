from model import *
from train_test_utils import * 
from train_attack_utils import * 
import copy
import random
import torch 
import numpy as np

class FedAvg:
    def __init__(self, clients_num ,device,client_epoch,non_idd,model,optimizer,lr,perturb_rate,attack,attack_fixed,eps): # def __init__(self, args):
        self.client_number = clients_num
        self.device = device
        self.client_epoch = client_epoch
        self.non_idd = non_idd
        self.optimizer = optimizer
        self.lr = lr
        self.perturb_rate = perturb_rate
        self.attack = attack
        self.attack_fixed = attack_fixed
        self.eps = eps

        self.clients_list = []  #10 clients
        for i in range(self.client_number):
            self.clients_list.append("client_"+str(i))
        
        # 指定模型形式
        if model == "ANN":
            self.n_features = 2
            self.nn =  ANN_Model(self.n_features) #0914 加入model
        else:
            self.nn =  small_ANN()
        self.nns = []

        for i in range(self.client_number):
            temp = copy.deepcopy(self.nn)
            #temp.name = self.clients_list[i] #不知道這是啥小
            self.nns.append(temp)
        
        
        # 固定幾個client被攻擊
        # if self.attack != "none":
        # print(f"perturb　rate:　{self.perturb_rate}")
        index_list = list(range(self.client_number))
        self.perturb_list  =  random.sample(index_list, round(self.perturb_rate*self.client_number))
        print("本次受攻擊的client為:",self.perturb_list)


    def server(self, r, client_rate , dataset):
        self.r = r  #the number of communication rounds
        self.client_rate = client_rate
        self.round_r2_list = list()
        

   
        # communication round從這裡開始啦!
        for t in range(self.r):
            print('round', t + 1, ':') #最外圈for loop
            # sampling
            m = np.max([int(self.client_rate * self.client_number), 1])
            index = random.sample(range(0, self.client_number), m)  #St #random set of m clients
            
            # 1. dispatch
            self.dispatch(index)

            # 2. local updating
            self.client_update(index,self.device,self.non_idd,self.eps)
           
            # 3. aggregation
            self.aggregation(index)

        return self.nn


    #1.
    def dispatch(self, index):
        print(index) 

        #print(self.nns[0].len) #從這裡改

        #將 self.nn 中的參數的值賦值給 self.nns 中的所有模型的參數。
        for j in index:
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone()
                #print(old_params.data)

    #2. 
    def client_update(self, index, device,non_iid,eps):  # update nn
        # print("開始更新個別client")
        self.device = device
        total_clients = self.client_number
        E = self.client_epoch
        optimizer = self.optimizer
        lr = self.lr 
        # eps = self.eps
        
        # 每個communicatin round 隨機被攻擊
        if self.attack_fixed == "non-fixed":
            print(f"perturb　rate:　{self.perturb_rate}")
            index_list = list(range(self.client_number))
            self.perturb_list  =  random.sample(index_list, round(self.perturb_rate*self.client_number))
            print("本次受攻擊的client為:",self.perturb_list )

        
        #訓練local模型
        for k in index:
            if k in self.perturb_list and self.attack == "fgsm" : 
                print("The client_{} start to train (FGSM) ".format(k))
                self.nns[k] = train_fgsm(self.nns[k],non_iid,lr,optimizer_name= optimizer,client_epoches = E, trained_clinet_number=k,total_clients= total_clients,eps = eps)
            
            elif k in self.perturb_list and self.attack == "pgd" :
                print("The client_{} start to train (PGD) ".format(k))
                self.nns[k] = train_pgd(self.nns[k],non_iid,lr,optimizer_name= optimizer,client_epoches = E, trained_clinet_number=k,total_clients= total_clients,eps = eps)
        
            else:
                print("The client_{} start to train(clean)".format(k))
                self.nns[k] = train(self.nns[k],non_iid,lr,optimizer_name= optimizer,client_epoches = E, trained_clinet_number=k,total_clients= total_clients)
        
       
        
    #3. 要想辦法理解!!!
    def aggregation(self, index): #question
        
        s = 0
        for j in index:
            # normal
            s += self.nns[j].len #s: 總資料量
            print(f"local_{j}:總資料量(dataloade)_{self.nns[j].len}")

        params = {}
        for name,  param in self.nns[0].named_parameters(): # model的名字與参数迭代器
            params[name] = torch.zeros_like(param.data)   

        for j in index:
            for name, param in self.nns[j].named_parameters(): 
                params[name] += param.data * (self.nns[j].len / s)
                
        for k, v in self.nn.named_parameters():
            v.data = params[k].data.clone()
        
        print(self.nns)
        print(len(self.nns))
        #key: 聚合後的模型: self.nn
        every_round_r2 = self.every_round_test(self.nn) #回傳測試的r2_score

        self.round_r2_list.append(every_round_r2)

        #把loss值寫出來另外保存
        f = open("every_round_r2.txt", "a")
        f.write(str(self.round_r2_list)+"\n")
        f.write("")
        f.close()
    
    #每一次溝通round的r2_score
    def every_round_test(self,agg_model):
        every_round_model = copy.deepcopy(agg_model)
        every_round_model.eval()
        every_round_r2 = test(every_round_model)
        return every_round_r2


    def server_test(self):
        model = self.nn
        server_model = copy.deepcopy(model)
        server_model.eval()

        server_name = ["server"]
        
        for server in server_name:
            server_model.name = server
            print("server測試結果為: ")
            test_r2_score = test(model)
        
        return model, test_r2_score
        

    #測試個別client的數據
    def global_test(self,clients_num):
        model = self.nn
        model.eval()
        clients_wind = ['Task1_W_Zone' + str(i) for i in range(1, clients_num)]
        c = clients_wind
        for client in c:
            model.name = client
            test(model)



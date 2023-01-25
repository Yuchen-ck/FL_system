from preprocessing import *
from torch.optim.lr_scheduler import StepLR
from torch import nn
from sklearn.metrics import r2_score

import copy

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

def train(model,non_iid,lr,optimizer_name,client_epoches,trained_clinet_number,total_clients):
    model.train() #開啟訓練模式
    # print("training process")
    


    Batch_size = 16
    training_loader = load_train_dataloader(trained_clinet_number ,total_clients, Batch_size,non_iid) #B :client's batch size
    model.len = len(training_loader) 
    
    loss_function = nn.L1Loss()
    
    # lr = 0.1
        
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = 1e-4)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
                              
    lr_step = StepLR(optimizer, step_size=10, gamma=0.1)

    E = client_epoches
    # the number of client's training
    mean_train_loss = []
    for epoch in range(E):
        train_loss = []
        for (seq, label,index) in training_loader:
            #seq = seq.to(device)
            #label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            
        lr_step.step()

        # validation
        # 暫時不需要
        # val_loss ,val_r2_score = get_val_loss(model, val_loader) 

        # print('epoch {:03d} train_loss {:.8f} val_loss {:.8f} val_r2_score {:.8f}'.format(epoch+1, np.mean(train_loss), val_loss , val_r2_score))
        print('epoch {:03d} train_loss {:.8f} '.format(epoch+1, np.mean(train_loss)))
        mean_train_loss.append(np.mean(train_loss))

        #深複製模型
        model.train()
        best_model = copy.deepcopy(model)

    #印出每一輪的loss graph
    print("----------------------------")
    # plt_loss_graph(mean_train_loss,trained_clinet_number)

    return best_model #關鍵是一定要回傳model才能進行聚合
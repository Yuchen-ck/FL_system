from uv_preprocessing_ import *
from torch.optim.lr_scheduler import StepLR
from torch import nn
from sklearn.metrics import r2_score

import copy

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

def train_uv(model,non_iid,lr,optimizer_name,client_epoches,trained_clinet_number,total_clients):
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
        
        #寫入tensorbroad
        # writer.add_scalar('val loss', val_loss, epoch)
        # writer.add_scalar('val acc', val_r2_score, epoch)

        #深複製模型
        model.train()
        best_model = copy.deepcopy(model)

    #印出每一輪的loss graph
    print("----------------------------")
    # plt_loss_graph(mean_train_loss,trained_clinet_number)

    return best_model #關鍵是一定要回傳model才能進行聚合



# def test(model):
#     #這裡要用測試集
#     model.eval()
#     Batch_size = 50
#     test_dataset = load_test_dataloader(Batch_size)
#     test_r2_score = []

#     with torch.no_grad():
#         for (X_test, y_true,index) in test_dataset:
        
#             #seq = seq.to(device)
#             y_pred = model(X_test)
#             test_r2_score.append(r2_score(y_pred,y_true))

#     avg_r2 = sum(test_r2_score) /len(test_r2_score)
#     print(round(avg_r2,4))

#     return round(avg_r2,5)

'''
# def get_val_loss(model, val_loader):
#     model.eval() #開始驗證模式
#     loss_function = nn.L1Loss()
#     val_loss = []
#     val_r2_score =[]
#     with torch.no_grad():
#         for (seq, label) in val_loader:
#             # seq = seq.to(args.device)
#             # label = label.to(args.device)
#             y_pred = model(seq)
#             #1. loss
#             loss = loss_function(y_pred, label)
#             val_loss.append(loss.item())
#             #2. r2_score
#             val_r2_score_number = r2_score_calculator(y_pred, label)
#             val_r2_score.append(val_r2_score_number)

#     return np.mean(val_loss) , np.mean(val_r2_score)

# def plt_loss_graph(train_losses,trained_clinet_number):
#     import matplotlib.pyplot as plt
#     fig = plt.figure()
#     plt.plot(train_losses, label='training loss')
#     plt.title('Loss at the end of each epoch')
#     plt.legend()
#     fig.savefig('./local_loss/client{}'.format(trained_clinet_number))



# def r2_score_calculator(y_pred, label):
    
#     y_pred = y_pred.view(y_pred.size(0),-1)
#     label = label.view(label.size(0),-1)

#     #自已發明的制式寫法，要多熟悉
#     r2_score_list =[]
#     for i in range(y_pred.shape[0]):
#         r2_score_list.append(r2_score(y_pred[i],label[i]))

#     val_r2_score = sum(r2_score_list) /len(r2_score_list)
    

#     return val_r2_score
'''






    

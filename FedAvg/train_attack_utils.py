from preprocessing import *
from torch.optim.lr_scheduler import StepLR
from torch import nn
from sklearn.metrics import r2_score
import copy

def fgsm_attack(model, loss, images, labels, eps) :
    
    images.requires_grad = True
            
    outputs = model(images)
    
    model.zero_grad()
    cost = loss(outputs, labels)
    cost.backward()
    
    attack_images = images + eps*images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    
    return attack_images

def train_fgsm(model,non_iid,lr,optimizer_name,client_epoches,trained_clinet_number,total_clients,eps):
    model.train() #開啟訓練模式
    # print("training process")
    Batch_size = 16
    training_loader = load_train_dataloader(trained_clinet_number ,total_clients, Batch_size,non_iid) #B :client's batch size
    model.len = len(training_loader) 
    
    loss_function = nn.L1Loss()
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = 1e-4)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
                              
    lr_step = StepLR(optimizer, step_size=10, gamma=0.1)

    E = client_epoches
    # the number of client's training
    mean_train_loss = []

    #攻擊eps參數未來移到main.py控制
    # eps = 0.5
    
    for epoch in range(E):
        train_loss = []
        for (seq, label,index) in training_loader:
            # 先試看看加躁對於收斂的影響!
            inputs = fgsm_attack(model, loss_function, seq, label, eps)
            y_pred = model(inputs)
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

def pgd_attack(model, images, labels, alpha, eps=0.3, iters=10) : #iters=40
    
    loss = nn.L1Loss()
        
    ori_images = images.data
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images



def train_pgd(model,non_iid,lr,optimizer_name,client_epoches,trained_clinet_number,total_clients):
    model.train() #開啟訓練模式
    # print("training process")
    Batch_size = 16
    training_loader = load_train_dataloader(trained_clinet_number ,total_clients, Batch_size,non_iid) #B :client's batch size
    model.len = len(training_loader) 
    
    loss_function = nn.L1Loss()
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = 1e-4)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
                              
    lr_step = StepLR(optimizer, step_size=10, gamma=0.1)

    E = client_epoches
    # the number of client's training
    mean_train_loss = []

    #攻擊eps參數未來移到main.py控制
    alpha = 2/225
    for epoch in range(E):
        train_loss = []
        for (seq, label,index) in training_loader:
            # 先試看看加躁對於收斂的影響!
            # inputs = fgsm_attack(model, loss_function, seq, label, eps)
            inputs =  pgd_attack(model, seq, label,alpha)
            y_pred = model(inputs)
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


# # 高斯加躁
# def train_noise(model,non_iid,lr,optimizer_name,client_epoches,trained_clinet_number,total_clients):
#     model.train() #開啟訓練模式
#     # print("training process")
#     Batch_size = 16
#     training_loader = load_train_dataloader(trained_clinet_number ,total_clients, Batch_size,non_iid) #B :client's batch size
#     model.len = len(training_loader) 
    
#     loss_function = nn.L1Loss()
    
#     if optimizer_name == 'adam':
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = 1e-4)
#     else:
#         optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
                              
#     lr_step = StepLR(optimizer, step_size=10, gamma=0.1)

#     E = client_epoches
#     # the number of client's training
#     mean_train_loss = []
#     noise_sd = 0.1
#     for epoch in range(E):
#         train_loss = []
#         for (seq, label,index) in training_loader:
#             # 先試看看加躁對於收斂的影響!
#             inputs = seq
#             inputs = inputs + torch.randn_like(inputs) * noise_sd
#             y_pred = model(inputs)
#             loss = loss_function(y_pred, label)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             train_loss.append(loss.item())
            
#         lr_step.step()

#         # validation
#         # 暫時不需要
#         # val_loss ,val_r2_score = get_val_loss(model, val_loader) 

#         # print('epoch {:03d} train_loss {:.8f} val_loss {:.8f} val_r2_score {:.8f}'.format(epoch+1, np.mean(train_loss), val_loss , val_r2_score))
#         print('epoch {:03d} train_loss {:.8f} '.format(epoch+1, np.mean(train_loss)))
#         mean_train_loss.append(np.mean(train_loss))
        
#         #寫入tensorbroad
#         # writer.add_scalar('val loss', val_loss, epoch)
#         # writer.add_scalar('val acc', val_r2_score, epoch)

#         #深複製模型
#         model.train()
#         best_model = copy.deepcopy(model)

#     #印出每一輪的loss graph
#     print("----------------------------")
#     # plt_loss_graph(mean_train_loss,trained_clinet_number)

#     return best_model #關鍵是一定要回傳model才能進行聚合
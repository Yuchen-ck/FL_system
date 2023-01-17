# PGD Attack
# MNIST init
import torch
import torch.nn as nn
from sklearn.metrics import r2_score 

def pgd_attack(model, images, labels, alpha,eps=0.3, iters=40) : #iters=40
    
    loss = nn.CrossEntropyLoss()
        
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

def test_with_PGD(model,test_loader,alpha):
    model.eval()
    # with torch.no_grad(): #不能有這行!!!
    for data, labels in test_loader:
        perturbed_images = pgd_attack(model, data, labels,alpha)
        output_attack = model(perturbed_images)

    return output_attack, labels

def test_with_PGD_MNIST(model,test_loader,alpha):
    model.eval()
    # Accuracy counter
    correct = 0
    total = 0
    
    # with torch.no_grad(): #不能有這行!!!
    for  images, labels in test_loader:
      	#攻击样本
        perturbed_images = pgd_attack(model, images, labels,alpha)
        output_attack = model(perturbed_images)

        #預測狀況
        _, pre = torch.max(output_attack.data, 1)

        total += 1
        correct += (pre == labels).sum()
    
    return total , correct



def r2_score_calculator(y_pred, label):
    
    # print(y_pred.shape)
    y_pred = y_pred.view(y_pred.size(0),-1).cpu()
    label = label.view(label.size(0),-1).cpu()
    

    #自已發明的制式寫法，要多熟悉
    r2_score_list =[]
    
    for i in range(y_pred.shape[0]):
        # print(y_pred[i])
        # print(label[i])
        r2_score_list.append(r2_score(y_pred[i].detach() ,label[i].detach() ))

    val_r2_score = sum(r2_score_list) /len(r2_score_list)
    

    return val_r2_score

def plt_attack_and_acc(alpha_list,accuracy_with_attack,picName,picPath,y_label):
    import matplotlib.pyplot as plt
    alphaName = ["0", "1/225", "2/225", "3/225", "4/225", "5/225"]
    fig = plt.figure()
    plt.title(picName)

    plt.plot(alpha_list, accuracy_with_attack, "*-")
    
    plt.xticks(alpha_list,alphaName)
    plt.xlabel("alpha") # 横坐标描述
    plt.ylabel(y_label)# 纵坐标描述

    # 设置数字标签 # 參考資料: https://blog.csdn.net/xiami_tao/article/details/79167273
    for a, b in zip(alpha_list, accuracy_with_attack):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    
    fig.savefig(picPath)
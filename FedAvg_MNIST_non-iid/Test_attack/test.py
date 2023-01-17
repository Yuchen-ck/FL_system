import torch
from sklearn.metrics import r2_score 
from model import *
from preprocessing import *
from pgd_test_alpha import *

'''
# # PGD Attack

# #白盒攻擊
# def fgsm_attack(image, epsilons, data_grad):
    
#     #获取梯度的符号
#     sign_data_grad = data_grad.sign()
#    #在原始图片的基础上加上epsilon成符号
#     perturbed_image = image + epsilons*sign_data_grad
#    #将数值裁剪到0-1的范围内
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)

#     return perturbed_image
    

# def test_with_FGSM(model,test_loader,epsilon,criterion):
#     model.eval()
#     # Accuracy counter
    
#     adv_examples = []
#     error = 0

#     correct_with_attack = 0
#     error_with_attack = 0
#     # with torch.no_grad(): #不能有這行!!!
#     for data, target in test_loader:
    
#         data.requires_grad = True #需要保存梯度

#         output = model(data) #利用分类器区分结果
#         init_pred = torch.max(output.data, 1)[1] #获取最大概率的索引值
        
#         # print(init_pred.shape)
#         # print(target.shape)
        
        
#         for i in range(len(init_pred)):
#             #如果分類錯誤，則不攻擊該樣本
#             if  init_pred[i] != target[i]:
#                 error+=1
#                 continue
        
#         # #如果分類錯誤，則不攻擊該樣本 #改成tolist()
#         # if init_pred.tolist() != target.tolist(): 
#         #     error +=1
#         #     continue

#         loss = criterion(output, target) #計算loss值
#         model.zero_grad()
#         loss.backward()

#         #-------------開始攻擊-------------

#         # 抽取梯度值
#         data_grad = data.grad.data

#       	#攻击样本
#         perturbed_data = fgsm_attack(data, epsilon, data_grad)

#         # 再次分类
#         output_attack = model(perturbed_data)

#         final_pred = torch.max(output_attack.data, 1)[1] # 得到最终分类结果

        
#         for i in range(len(final_pred)):
#             if final_pred[i] == target[i]:   #如果最终分类结果和真实结果相同
#                 correct_with_attack += 1

#                 # # 保存epsilon为0 #完全看不懂
#                 # if (epsilon == 0) and (len(adv_examples) < 5):
#                 #     adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
#                 #     #adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
#                 #     print("這到底是啥")
#                 #     print(adv_ex)

#             else:
#                 error_with_attack += 1

#                 # # 保存其他的一些结果，用于后期的可视化操作 #完全看不懂
#                 # if len(adv_examples) < 5:
#                 #     adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
#                 #     #adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        
        
    
#     return error , correct_with_attack , error_with_attack


# # 其他計算包

# def plt_attack_and_acc(epsilons,accuracy_with_attack,picName,result_path):
#     import matplotlib.pyplot as plt
#     import numpy as np

#     fig = plt.figure()
#     plt.title(picName)

#     plt.plot(epsilons, accuracy_with_attack, "*-")
    
#     plt.xticks(np.arange(0, .35, step=0.05))
#     plt.xlabel("Epsilon") # 横坐标描述
#     plt.ylabel("Accuracy(%)")# 纵坐标描述

#     # 设置数字标签 # 參考資料: https://blog.csdn.net/xiami_tao/article/details/79167273
#     for a, b in zip(epsilons, accuracy_with_attack):
#         plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    
#     fig.savefig(result_path)

'''

if __name__ == '__main__':
    #load model
    model = CNN_Model() 
    model.load_state_dict(torch.load("../diff_clients_agg.pth"))
    print(model)
    criterion =  nn.CrossEntropyLoss()
    #load testing set
    batch_size = 1024
    testing_loader,len_of_test_dataset = load_MNIST_testing_set(batch_size)
    

    # #FGSM attack result
    # epsilons = [0, .05, .1, .15, .2, .25, .3]
    # accuracy_with_attack = []

    # #攻擊後的準確度
    # for eps in epsilons:

    #     print('{} is:'.format(eps))

    #     _ , acc_with_attack , error_with_attack = test_with_FGSM(model,testing_loader,eps,criterion)
    #     print('correct with attack: {}%'.format(acc_with_attack/len_of_test_dataset *100))
    #     acc = acc_with_attack/len_of_test_dataset *100
    #     accuracy_with_attack.append(round(acc,3))
        
    # picName = "FL_MNIST Accuracy vs. FGSM Epsilon"
    # result_path = './plt/FL_MNIST_FGSM_attack.png'

    # plt_attack_and_acc(epsilons,accuracy_with_attack,picName,result_path)


    #PGD attack result
    alpha_list = [0, 1/225, 2/225, 3/225, 4/225, 5/225]
    accuracy_with_attack = []

    #攻擊後的準確度
    for alpha in alpha_list:

        print('{} is:'.format(alpha))

        total , correct = test_with_PGD_MNIST(model,testing_loader,alpha)
        
        acc =  (float(correct) / total)
        print("{}%".format(acc/10))
        accuracy_with_attack.append(round(acc/10,3))
        
    y_label = "Accuracy (%)"
    picName = "FL_MNIST Accuracy vs. PGD Alpha"
    result_path = './plt/FL_MNIST_PGD_attack.png'

    plt_attack_and_acc(alpha_list,accuracy_with_attack,picName,result_path,y_label)

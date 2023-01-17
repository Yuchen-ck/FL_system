from preprocessing import *
from torch.optim.lr_scheduler import StepLR
from torch import nn
from sklearn.metrics import r2_score
import numpy as np
import copy


def train(model,non_iid,optimizer_name,client_epoches,trained_clinet_number,total_clients):
    model.train() #開啟訓練模式
    print("開啟訓練模式: ")
    print(trained_clinet_number)
    batch_size = 50
    train_loader = load_MNIST_training_set(batch_size,trained_clinet_number,total_clients,non_iid) #B :client's batch size
    model.len = len(train_loader) 
    criterion = nn.CrossEntropyLoss()
    
    lr = 0.01
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.9, weight_decay = 1e-4)
    # lr_step = StepLR(optimizer, step_size=10, gamma=0.1)

    E = client_epoches
    train_losses = []
    train_correct = []
    correct = 0
    total = 0

    for epoch in range(E):
        train_loss = []
        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):
            b+=1
    
            # Apply the model
            y_pred = model(X_train)  # we don't flatten X-train here
            loss = criterion(y_pred, y_train)
    
            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            accuracy = predicted.eq(y_train.data).sum()/ batch_size*100
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print interim results
            if b%10 == 0:
                print(f'epoch: {epoch:2}  batch: {b:4} [{10*b:6}/60000]  loss: {loss.item():10.8f}  accuracy: {accuracy:7.3f}%')

        print("------------next epoch------------")
        train_losses.append(loss.item())
        train_correct.append(predicted.eq(y_train.data).sum()/ batch_size*100)
        model.train()

        # print("複製模型前")
        best_model = copy.deepcopy(model)

    # #印出每一輪的loss graph
    # print("----------------------------")
    # plt_loss_graph(mean_train_loss,trained_clinet_number)

    return best_model #關鍵是一定要回傳model才能進行聚合


def test(model):
    #這裡要用測試集
    model.eval()
    batch_size = 50
    test_loader = load_MNIST_testing_set(batch_size)
    test_acc = []

    # Accuracy counter
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            
            output = model(data)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()
        
        print(len(test_loader.dataset))
    
    print('\nTesting without attack - Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))


def plt_loss_graph(train_losses,trained_clinet_number):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(train_losses, label='training loss')
    plt.title('Loss at the end of each epoch')
    plt.legend()
    fig.savefig('./local_loss/client{}'.format(trained_clinet_number))



from model import *
from preprocessing import *
from attack import *

import argparse

def get_args(): #控制攻擊狀態
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'none'])
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--attack-iters', default=40, type=int)
    parser.add_argument('--alpha', default=1e-2, type=float)
    parser.add_argument('--restarts', default=10, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    #load model
    model = CNN_Model() 
    model.load_state_dict(torch.load("./2023-01-20_91.39999389648438.pth"))
    # model.load_state_dict(torch.load("./2023-01-19_97.73999786376953.pth"))
    print(model)
    
    #load test_loader
    batch_size = 100
    test_loader ,len_of_test_dataset = load_MNIST_testing_set(batch_size)
    
    # testing stage 
    model.eval()
    total_loss = 0
    total_acc = 0
    n = 0

    if args.attack == 'none':
        with torch.no_grad():
            for i, (X, y) in enumerate(test_loader):
                # X, y = X.cuda(), y.cuda()
                output = model(X)
                loss = F.cross_entropy(output, y)
                total_loss += loss.item() * y.size(0)
                total_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
    else:
        for i, (X, y) in enumerate(test_loader):
            # X, y = X.cuda(), y.cuda()
            
            if args.attack == 'pgd':
                # X, y = X.cuda(), y.cuda()
                # for attack_iters in range(10,50,10):
                delta = attack_pgd(model, X, y, args.epsilon, args.alpha, args.attack_iters, args.restarts)
                print("pgd")
            
            elif args.attack == 'fgsm':
                delta = attack_fgsm(model, X, y, args.epsilon)
                print("fgsm")

            with torch.no_grad():
                output = model(X + delta)
                loss = F.cross_entropy(output, y)
                total_loss += loss.item() * y.size(0)
                total_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)

    print(f'Test Loss: {total_loss/n}, Acc: {total_acc/n}')
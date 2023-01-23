import torch
def FGSM_attack(X,y,model,criterion,epsilon,alpha):
    delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda() #在[-args.epsilon, args.epsilon]的隨機均匀分布中取值，並重新赋值
    delta.requires_grad = True
    output = model(X + delta)
    loss = criterion(output, y)
    loss.backward()
    grad = delta.grad.detach()
    delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
    delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
    delta = delta.detach()
    return delta

def PGD_attack(X,y,model,criterion,opt,epsilon,attack_iters):
    delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
    delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
    for _ in range(attack_iters):
        delta.requires_grad = True
        output = model(X + delta)
        loss = criterion(output, y)
        opt.zero_grad()
        loss.backward()
        grad = delta.grad.detach()
        
    delta = delta.detach()
    return delta
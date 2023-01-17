from torch import nn
import torch
def fgsm_attack(model, loss, images, labels, eps) :
    
    images.requires_grad = True
            
    outputs = model(images)
    
    model.zero_grad()
    cost = loss(outputs, labels)
    cost.backward()
    
    attack_images = images + eps*images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    
    return attack_images

def pgd_attack(model, images, labels, alpha, eps=0.3, iters=10) : #iters=40
    
    loss = nn.L1Loss()
        
    ori_images = images.data
        
    for i in range(iters):    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images

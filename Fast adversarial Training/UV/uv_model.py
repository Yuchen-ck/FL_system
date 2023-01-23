import torch
import torch.nn as nn
from torch.nn import functional as F

# 
class small_ANN(nn.Module):
  def __init__(self,n_features):
    super(small_ANN, self).__init__()
    self.fc1 = nn.Linear(n_features, 5)  # 2-(10-10)-1
    self.fc3 = nn.Linear(5, 1)

    nn.init.xavier_uniform_(self.fc1.weight)  # glorot
    nn.init.zeros_(self.fc1.bias)
   
    nn.init.xavier_uniform_(self.fc3.weight)
    nn.init.zeros_(self.fc3.bias)

  def forward(self, x):
    z = F.relu(self.fc1(x))  # or T.nn.Tanh()
    z = self.fc3(z)  # no activation, aka Identity()
    return z

import torch.nn as nn
import torch.nn.functional as F
import torch

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.fc1 = nn.Linear(3, 15)  # 2-(10-10)-1
    self.fc2 = nn.Linear(15, 25)
    self.fc3 = nn.Linear(25, 5)
    self.fc4 = nn.Linear(5, 1)



  def forward(self, x):
    z = F.relu(self.fc1(x))  # or T.nn.Tanh()
    z = F.relu(self.fc2(z))
    z = F.relu(self.fc3(z))
    z = self.fc4(z)  # no activation, aka Identity()
    return z
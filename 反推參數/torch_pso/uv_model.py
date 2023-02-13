
from torch import nn
from torch.nn import functional as F


class small_ANN(nn.Module):
  def __init__(self):
    super(small_ANN, self).__init__()
    self.fc1 = nn.Linear(3, 5)  # 2-(10-10)-1
    self.fc3 = nn.Linear(5, 1)

    nn.init.xavier_uniform_(self.fc1.weight)  # glorot
    nn.init.zeros_(self.fc1.bias)
   
    nn.init.xavier_uniform_(self.fc3.weight)
    nn.init.zeros_(self.fc3.bias)

  def forward(self, x):
    z = F.relu(self.fc1(x))  # or T.nn.Tanh()
    z = self.fc3(z)  # no activation, aka Identity()
    return z
  

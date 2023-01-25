
from torch import nn
from torch.nn import functional as F

class ANN_Model(nn.Module):
    def __init__(self, n_features):
        super(ANN_Model, self).__init__()
        self.linearA = nn.Linear(n_features, 10)
        self.linearB = nn.Linear(10, 16)
        self.linearC = nn.Linear(16, 4)
        self.linearD = nn.Linear(4, 1)

    def forward(self, x):
        X = F.relu(self.linearA(x))
        X = F.relu(self.linearB(X))
        X= F.relu(self.linearC(X))
        return self.linearD(X)


class small_ANN(nn.Module):
  def __init__(self):
    super(small_ANN, self).__init__()
    self.fc1 = nn.Linear(2, 5)  # 2-(10-10)-1
    self.fc3 = nn.Linear(5, 1)

    nn.init.xavier_uniform_(self.fc1.weight)  # glorot
    nn.init.zeros_(self.fc1.bias)
   
    nn.init.xavier_uniform_(self.fc3.weight)
    nn.init.zeros_(self.fc3.bias)

  def forward(self, x):
    z = F.relu(self.fc1(x))  # or T.nn.Tanh()
    z = self.fc3(z)  # no activation, aka Identity()
    return z
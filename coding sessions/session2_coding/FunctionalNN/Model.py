from torch import nn
import torch.nn.functional as F

from settings import D_in, H, D_out

class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(D_in,H)
        self.fc2 = nn.Linear(H,D_out)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
import torch.nn as nn
import torch.nn.functional as F
import torch


class Linear(nn.Module) :
    def __init__(self, input_size, hidden_size, num_classes):
        super(Linear, self).__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, text):
        x = self.fc1(text)
        x = self.fc2(x)
        preds = self.fc3(x)  # find the predictions
        return preds




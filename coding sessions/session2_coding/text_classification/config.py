import torch
import torch.nn as nn
import os

# datataset
train_path = 'data/train.tsv'
test_path = 'data/test.tsv'
split_ratio = 0.8
labels_mapper = {'OFF': 1, 'NOT': 0}

# model
num_classes = 2  # number of classes
hidden_size = 100
max_length = 50

# training hyperparameters
lr = 0.001
loss_func = nn.CrossEntropyLoss()
num_epochs = 20
batch_size = 64

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# where to store results
results = "./results"
if not os.path.exists(results):
    os.makedirs(results)

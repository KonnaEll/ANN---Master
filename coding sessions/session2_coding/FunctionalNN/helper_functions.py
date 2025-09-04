from settings import Epochs, batch_size, LR
from torch import nn
from torch import optim
import numpy as np
from matplotlib import pyplot as plt

def train(model, train_loader, validation_loader):
    # defining cross entropy loss
    criterion = nn.CrossEntropyLoss()
    # creating an optimizer object performing SGD algorithm
    optimizer = optim.SGD(model.parameters(),lr=LR)
    train_loss_list_per_epoch = []
    train_loss_list_per_itr = []
    val_loss_list = []
    for epoch in range(Epochs):
        itr = 0
        for inputs, labels in train_loader:

            outputs = model(inputs.view(batch_size,-1))
            loss = criterion(outputs,labels)

            # clearing old gradients from the last step
            optimizer.zero_grad()
            # computing the derivative of the loss w.r.t. the parameters
            loss.backward()
            # optimizer takes a step in updating parameters based on the gradients of the parameters.
            optimizer.step()
            if itr % 10 ==0:
                train_loss_list_per_itr.append(loss.item())
            itr += 1
        train_loss_list_per_epoch.append(np.mean(train_loss_list_per_itr))
        # Evaluate model for each update iteration
        eval_loss = evalution(model, validation_loader, criterion)
        val_loss_list.append(eval_loss)

    # plotting the loss curve over all iteration
    print('Train loss values per iteration')
    print(train_loss_list_per_epoch)
    plt.plot(np.arange(len(train_loss_list_per_epoch)), train_loss_list_per_epoch,color='blue',label='Train')
    plt.plot(np.arange(len(val_loss_list)), val_loss_list,color='red',label='Validation')
    plt.legend()
    plt.show()
    plt.cla()

def evalution(model, validation_loader,criterion):
    val_loss = []
    for inputs, labels in validation_loader:
        outputs = model(inputs.view(batch_size, -1))
        loss = criterion(outputs, labels)
        val_loss.append(loss.item())
    return np.mean(val_loss)
import torch
from time import time
import matplotlib.pyplot as plt
import os

from config import *

def accuracy(probs, target):
    winners = probs.argmax(dim=1)
    corrects = (winners == target)
    accuracy = corrects.sum().float() / float(target.size(0))
    return accuracy


def train(model, dataloader, optimizer, criterion, device):
    timer_start = time()
    epoch_loss = 0
    epoch_acc = 0
    model = model.to(device)
    for batch in dataloader:
        text, label = batch
        label = label.type(torch.LongTensor)
        # zero the parameter gradients
        optimizer.zero_grad()
        # retrieve text and label and calculate loss and acc based on predictions
        predictions = model(text.to(device))
        loss = criterion(predictions.to(device), label.squeeze())
        acc = accuracy(predictions.to(device), label)
        # perform backpropagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    print(f"Training time for this epoch {time()-timer_start:.2f}")
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model = model.to(device)
    model.eval()
    gt_labels = []
    all_predictions = []
    with torch.no_grad():
        for text, label in dataloader:
            label = label.type(torch.LongTensor)
            predictions = model(text.to(device))
            loss = criterion(predictions.to(device), label.squeeze())
            acc = accuracy(predictions.to(device), label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            gt_labels += label.tolist()
            all_predictions += torch.argmax(predictions, axis=1).tolist()
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def run_train(epochs, model, train_dataloader, valid_dataloader, optimizer, criterion, device):
    best_valid_loss = float("inf")
    train_loss_list, train_acc_list, valid_loss_list, valid_acc_list = list(), list(), list(), list()
    for epoch in range(epochs):
        # train the model
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device)
        # evaluate the model
        valid_loss, valid_acc = evaluate(model, valid_dataloader, criterion, device)
        # storing loss and accuracy values for both training and validation
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)
        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(results, "best_weights.pt"))
        print(f"\tEPOCH {epoch} Train Loss : {train_loss:.3f} | Train Acc : {train_acc * 100:.2f}%")
        print(f"\tEPOCH {epoch} Val Loss : {valid_loss:.3f} | Val Acc : {valid_acc * 100:.2f}%")

    """#Plot Training & Validation Loss and Acc"""
    epochs_list = torch.arange(epochs)
    plt.plot(epochs_list, train_acc_list, label='Training acc')
    plt.plot(epochs_list, valid_acc_list, label='Validation acc')
    plt.title('Model Acc')
    plt.ylabel('Acc')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig(os.path.join(results, "acc.png"))

    plt.figure()
    plt.plot(epochs_list, train_loss_list, label='Training loss')
    plt.plot(epochs_list, valid_loss_list, label='Validation loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig(os.path.join(results, "loss.png"))

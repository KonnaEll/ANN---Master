from dataset import create_dataset, data_split, tsv_to_list
from helper import run_train, evaluate
from model import Linear
from config import *
from torch.utils.data import TensorDataset, DataLoader


# reading tsv file into list of texts and labels
train_text, train_labels = tsv_to_list(train_path)
# split train data into train and validation
train_text, train_labels, val_text, val_labels = data_split(train_text, train_labels, split_ratio)
# apply defined preprocessing steps on train samples
train_text, train_labels, vocabs = create_dataset(train_text, train_labels, max_length=max_length)
# apply defined preprocessing steps on val samples
val_text, val_labels, vocabs = create_dataset(val_text, val_labels, vocabs=vocabs, max_length=max_length)

# reading tsv file into list of texts and labels
test_texts, test_labels = tsv_to_list(test_path)
# apply defined preprocessing steps on test samples
test_texts, test_labels, vocabs = create_dataset(test_texts, test_labels, vocabs=vocabs, max_length=max_length)

# create Pytorch Dataset to be feed into Dataloader
train_dataset = TensorDataset(train_text, train_labels)
val_dataset = TensorDataset(val_text, val_labels)
test_dataset = TensorDataset(test_texts, test_labels)

# create dataloaders based on datasets
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

#  Building the model
linear_model = Linear(max_length, hidden_size, num_classes)
linear_model = linear_model.to(device)

# creating an optimizer
pytorch_total_params = sum(p.numel() for p in linear_model.parameters() if p.requires_grad)
print("Total number of trainable parameters: ", pytorch_total_params)
optimizer = torch.optim.Adam(linear_model.parameters(), lr=lr)

# training the model
run_train(num_epochs, linear_model, train_dataloader, val_dataloader, optimizer, loss_func, device)

# Evaluate the trained model
test_loss, test_acc = evaluate(linear_model, test_dataloader, loss_func, device)
print(f"Test Loss : {test_loss:.3f} | Test Acc : {test_acc * 100:.2f}%")




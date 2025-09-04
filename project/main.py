import torch
import os
import pandas as pd
import numpy as np
import warnings
from functions import read_config, CustomDataset, CustomBlurDataset, CustomPerturbedDataset, train_model, validate_model, evaluate_test
from torchvision import models
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import random
from PIL import Image

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seed = 42
set_seed(same_seed)

warnings.filterwarnings('ignore')

config = read_config('config.txt')
epochs = int(config['epochs'])
data_path = config['data_path']
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64]
count_folders = len([name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))])

transform = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
entropyloss = torch.nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################                     FULLY SUPERVISED MODEL                       #################################
print("FULLY SUPERVISED MODEL")

scene_data = CustomDataset(img_dir=data_path, transform=transform) # load data from the folders into scene_data

# Obtain labels for each index
indices = list(range(len(scene_data)))
labels = [scene_data[i][1] for i in indices]

# Stratified split using scikit-learn's train_test_split
train_set_idx, test_set_idx, _, _ = train_test_split(indices, labels, test_size=0.2, random_state=same_seed, stratify=labels)

# Create subset data loaders
train_dataset = Subset(scene_data, train_set_idx)
test_dataset = Subset(scene_data, test_set_idx)

# Show one random picture from each img_folder_path with the label
folder_paths = [os.path.join(data_path, folder) for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))]
fig, axs = plt.subplots(3, 5, figsize=(10, 10))  # Adjusted for 5 rows and 3 columns
for i, folder in enumerate(folder_paths):
    label = os.path.basename(folder)
    img_path = random.choice(os.listdir(folder))
    img = Image.open(os.path.join(folder, img_path)).convert('RGB')
    img = img.resize((224, 224))
    img_np = np.array(img)

    # Show image
    row, col = divmod(i, 5)
    axs[row, col].imshow(img_np)
    axs[row, col].set_title(f"Label: {label}")
    axs[row, col].axis('off')

plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Add space between plots
plt.tight_layout()
plt.savefig('original_images.png')


# Iterate through the combinations of learning rates and batch sizes
results = {}
best_epochs = {}
for lr in learning_rates:
    for bs in batch_sizes:
        log_file = f'fully_supervised_{lr}_and_{bs}.txt'
        print(f"Training with learning rate: {lr}, batch size: {bs}")

        set_seed(same_seed)
        # load the datasets
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

        # Load pre-trained EfficientNet-B0
        efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_features = efficientnet.classifier[1].in_features
        efficientnet.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2), # Add dropout layer to reduce overfitting
            torch.nn.Linear(num_features, count_folders) # Define a new fully connected layer with 15 output features
        )
        efficientnet.to(device)

        # Make all layers trainable
        for param in efficientnet.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(efficientnet.parameters(), lr=lr)  # initialize optimizer

        # Train and validate the model
        history, best_model_epoch = train_model(efficientnet, train_loader, test_loader, epochs, optimizer, entropyloss, log_file)

        results[(lr, bs)] = history
        best_epochs[(lr, bs)] = best_model_epoch

# Find the best model based on validation accuracy and keep the parameters
best_params = max(results, key=lambda k: max(results[k]['val_accuracy']))
best_history = results[best_params]
best_lr, best_bs = best_params
best_epoch = best_epochs[best_params]
set_seed(same_seed)

print(f"Best model found with learning rate: {best_lr}, batch size: {best_bs}")

# Plot the training and validation loss for the best model
plt.figure(figsize=(10, 5))
plt.plot(best_history['train_loss'], label='Train Loss')
plt.plot(best_history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for Best Model')
plt.legend()
plt.savefig('best_model_loss_plot.png')
plt.show()

# Load the best model with exaclty the same way
train_loader = DataLoader(train_dataset, batch_size=best_bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_bs, shuffle=False)

best_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
num_features = best_model.classifier[1].in_features
best_model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(num_features, count_folders)
)
best_model.to(device)

# Make all layers trainable
for param in best_model.parameters():
    param.requires_grad = True

# Reinitialize the optimizer for the best model
optimizer = torch.optim.Adam(best_model.parameters(), lr=best_lr)

log_file = f'fully_supervised_best.txt'
# Train the best model again
_, _ = train_model(best_model, train_loader, test_loader, best_epoch, optimizer, entropyloss, log_file)

# Call the detailed evaluation function
test_file = f'fully_supervised_testing.txt'
evaluate_test(best_model, test_loader, entropyloss, test_file)



# ###################################################              BLURRING             ###########################################
print("BLURRED MODEL")
log_file = f'blurring_validation.txt'
sec_log_file = f'after_blurring_validation.txt'
test_file = f'blurring_testing.txt'

# Create the dataset
blurred_data = CustomBlurDataset(img_dir=data_path, transform=transform)

indices = list(range(len(blurred_data)))
labels = [blurred_data[i][1] for i in indices]

# Stratified split using scikit-learn's train_test_split
blurred_train_set_idx, blurred_test_set_idx, _, _ = train_test_split(indices, labels, test_size=0.2, random_state=same_seed, stratify=labels)

# Create subset data loaders
blurred_train_dataset = Subset(blurred_data, blurred_train_set_idx)
blurred_test_dataset = Subset(blurred_data, blurred_test_set_idx)
blurred_train_loader = DataLoader(blurred_train_dataset, batch_size=best_bs, shuffle=True)
blurred_test_loader = DataLoader(blurred_test_dataset, batch_size=best_bs, shuffle=False)

# load pre-trained EfficientNet-B0
efficientnet_blur = models.efficientnet_b0(weights=True, pretrained=True)

# Modify the classifier to predict the blur kernel size
num_features = efficientnet_blur.classifier[1].in_features
efficientnet_blur.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(num_features, 5)  # 5 classes for the 5 kernel sizes
)
efficientnet_blur.to(device)

# make all layers trainable
for param in efficientnet_blur.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(efficientnet_blur.parameters(), lr=best_lr)

set_seed(same_seed)
_, _ = train_model(model=efficientnet_blur, dataloader=blurred_train_loader, validation_loader=blurred_test_loader, epochs=best_epoch, optimizer=optimizer, entropyloss=entropyloss, log_file=log_file)

# Modify the classifier for the main task
num_features = efficientnet_blur.classifier[1].in_features
efficientnet_blur.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(num_features, count_folders)  # count_folders is calculated as the number of folders in the directory we are working
)
efficientnet_blur.to(device)

# Freeze feature extraction part
for param in efficientnet_blur.features.parameters():
    param.requires_grad = False
for param in efficientnet_blur.classifier.parameters():
    param.requires_grad = True

# Reinitialize the optimizer for fine-tuning
optimizer = torch.optim.Adam(efficientnet_blur.parameters(), lr=best_lr)
set_seed(same_seed)

# Fine-tune the model on the main task
_, _ = train_model(model=efficientnet_blur, dataloader=train_loader, validation_loader=test_loader, epochs=best_epoch, optimizer=optimizer, entropyloss=entropyloss, log_file=sec_log_file)
evaluate_test(efficientnet_blur, test_loader, entropyloss, log_file=test_file)



###########################################                     PERTURBATION                       #########################################
print("PERTURBED MODEL")
log_file = f'perturbation_validation.txt'
sec_log_file = f'after_perturbation_validation.txt'
test_file = f'perturbation_testing.txt'

# Create the dataset
perturbed_data = CustomPerturbedDataset(img_dir=data_path, transform=transform)

indices = list(range(len(perturbed_data)))
labels = [perturbed_data[i][1] for i in indices]

# Stratified split using scikit-learn's train_test_split
perturbed_train_set_idx, perturbed_test_set_idx, _, _ = train_test_split(indices, labels, test_size=0.2, random_state=same_seed, stratify=labels)

# Create subset data loaders
perturbed_train_dataset = Subset(perturbed_data, perturbed_train_set_idx)
perturbed_test_dataset = Subset(perturbed_data, perturbed_test_set_idx)
perturbed_train_loader = DataLoader(perturbed_train_dataset, batch_size=best_bs, shuffle=True)
perturbed_test_loader = DataLoader(perturbed_test_dataset, batch_size=best_bs, shuffle=False)

# load pre-trained EfficientNet-B0
efficientnet_pert = models.efficientnet_b0(weights=True, pretrained=True)
num_features = efficientnet_pert.classifier[1].in_features
efficientnet_pert.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(num_features, 2)  # 2 classes one for white and one for black
)
efficientnet_pert.to(device)

# make all layers trainable
for param in efficientnet_pert.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(efficientnet_pert.parameters(), lr=best_lr)
set_seed(same_seed)
_, _ = train_model(model=efficientnet_pert, dataloader=perturbed_train_loader, validation_loader=perturbed_test_loader, epochs=best_epoch, optimizer=optimizer, entropyloss=entropyloss, log_file=log_file)

num_features = efficientnet_pert.classifier[1].in_features
efficientnet_pert.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(num_features, count_folders) # count_folders is calculated as the number of folders in the directory we are working
)
efficientnet_pert.to(device)

# Freeze feature extraction part
for param in efficientnet_pert.features.parameters():
    param.requires_grad = False
for param in efficientnet_pert.classifier.parameters():
    param.requires_grad = True

# train for the scene classification and evaluate
optimizer = torch.optim.Adam(efficientnet_pert.parameters(), lr=best_lr)
set_seed(same_seed)
_, _ = train_model(model=efficientnet_pert, dataloader=train_loader, validation_loader=test_loader, epochs=best_epoch, optimizer=optimizer, entropyloss=entropyloss, log_file=sec_log_file)
evaluate_test(efficientnet_pert, test_loader, entropyloss, log_file=test_file)
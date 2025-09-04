import torch
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import cv2
import random
from torch.utils.data import Dataset
from PIL import Image

def read_config(file_path): # read configuration file
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            config[key.strip()] = value.strip()
    return config

class CustomDataset(Dataset): # read dataset for fully supervised model
    def __init__(self, img_dir, transform):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for label in os.listdir(self.img_dir): # go to the folders and read each file
            img_folder_path = os.path.join(self.img_dir, label)
            for img in os.listdir(img_folder_path):
                img_path = os.path.join(img_folder_path, img)
                self.images.append(img_path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx): # open image and get the image and its label which is the folder name
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class CustomBlurDataset(Dataset): # same for blurring task
    def __init__(self, img_dir, transform):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.kernel_sizes = [(5, 5), (9, 9), (13, 13), (17, 17), (21, 21)]

        for label in os.listdir(self.img_dir):
            img_folder_path = os.path.join(self.img_dir, label)
            for img in os.listdir(img_folder_path):
                img_path = os.path.join(img_folder_path, img)
                self.images.append(img_path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.images) * len(self.kernel_sizes) # 5 kernels

    def __getitem__(self, idx): # open image and based on each kernel size make it blurred
        img_idx = idx // len(self.kernel_sizes)
        kernel_idx = idx % len(self.kernel_sizes)

        image = Image.open(self.images[img_idx]).convert('RGB')
        kernel_size = self.kernel_sizes[kernel_idx]

        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        blurred_image = cv2.GaussianBlur(image_np, kernel_size, sigmaX=0)
        blurred_image = Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))

        if self.transform:
            blurred_image = self.transform(blurred_image)

        label = kernel_idx
        return blurred_image, label

class CustomPerturbedDataset(Dataset):
    def __init__(self, img_dir, transform):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for label in os.listdir(self.img_dir):
            img_folder_path = os.path.join(self.img_dir, label)
            for img in os.listdir(img_folder_path):
                img_path = os.path.join(img_folder_path, img)
                self.images.append(img_path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.images) * 2  # Each image will have 2 perturbations

    def __getitem__(self, idx): # same as blurring, take image and make it black or white, but make sure the square is not out of the image
        img_idx = idx // 2
        perturbation_type = idx % 2

        image = Image.open(self.images[img_idx]).convert('RGB')
        image_np = np.array(image)

        # Apply perturbation
        perturb_value = 0 if perturbation_type == 0 else 255
        x = random.randint(0, image_np.shape[1] - 11)
        y = random.randint(0, image_np.shape[0] - 11)

        image_np[y:y+10, x:x+10] = perturb_value

        perturbed_image = Image.fromarray(image_np)

        if self.transform:
            perturbed_image = self.transform(perturbed_image)

        label = perturbation_type
        return perturbed_image, label

def train_model(model, dataloader, validation_loader, epochs, optimizer, entropyloss, log_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n-----------   Model training   -----------\n")
    best_accuracy = 0
    size_loader = len(dataloader)
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()  # set model to training mode
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            prediction = model(images) # predicted outputs
            loss = entropyloss(prediction, labels) # entropy loss

            optimizer.zero_grad() # reset the gradients of model parameters
            loss.backward() # backpropagate the prediction loss
            optimizer.step() # adjust the parameters by the gradients collected in the backward pass

            running_loss += loss.item()

        epoch_loss = running_loss / size_loader
        validation_loss, validation_accuracy = validate_model(model, validation_loader, entropyloss) # validate the model and keep the metrics
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(validation_loss)
        history['val_accuracy'].append(validation_accuracy)
        with open(log_file, 'a') as f:
            f.write(f'Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}, Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.4%}\n')

        # Save the model's metrics if it has the best accuracy so far
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_model_epoch = epoch + 1

    with open(log_file, 'a') as f:
        f.write(f'\nBest Accuracy: {best_accuracy:.2%} at epoch {best_model_epoch}\n')
    return history, best_model_epoch


def validate_model(model, validation_loader, entropyloss):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n-----------   Model validation   -----------\n")
    model.eval()  # Set model to evaluation mode
    validation_loss = 0.0
    correct = 0
    total = 0
    size_loader = len(validation_loader)

    with torch.no_grad(): # find validation loss and validation accuracy
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            prediction = model(images)
            validation_loss += entropyloss(prediction, labels).item()

            correct += (prediction.argmax(1) == labels).type(torch.float).sum().item()
            total += labels.size(0)

    validation_loss /= size_loader
    validation_accuracy = correct / total
    return validation_loss, validation_accuracy


def evaluate_test(model, dataloader, entropyloss, log_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n-----------   Test evaluation   -----------\n")
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    size_loader = len(dataloader)

    with torch.no_grad(): # find the metrics for the evaluation of the model to see how well the model works
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = entropyloss(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / size_loader
    accuracy = accuracy_score(all_labels, all_predictions)
    with open(log_file, 'a') as f:
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4%}\n")
        f.write(classification_report(all_labels, all_predictions))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(all_labels, all_predictions)))


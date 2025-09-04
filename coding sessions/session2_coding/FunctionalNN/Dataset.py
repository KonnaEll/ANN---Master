from torchvision import transforms, datasets
import torch

from settings import batch_size, mean_std_normalization

def load_date():
    # Creating a transformers including all pre-processing components such as normalization.
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean_std_normalization[0],
                                                         mean_std_normalization[1])])

    # downloading CIFAR10 dataset and sotring it in /data folder in the current directory
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # we just go for a subset of images
    subset = list(range(0, len(trainset), 5))
    trainset = torch.utils.data.Subset(trainset, subset)
    print('number of train examples ',len(trainset))
    validationset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    subset = list(range(0, len(trainset), 5))
    validationset = torch.utils.data.Subset(validationset, subset)
    print('number of evaluation examples ',len(validationset))
    # Creating a loader object to read and load a batch of data
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(validationset, batch_size=batch_size, shuffle=False, num_workers=0)

    # in case you need the name of classes in this dataset, we have provided a tuple of class namess
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, validation_loader

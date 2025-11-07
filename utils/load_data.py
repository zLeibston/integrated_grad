import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

def load_MNIST(root="./data/MNIST",isTrain=True):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST(root=root, train=isTrain, transform=transform, download=False)

    if(isTrain):
        train_size = int(0.9 * len(dataset)) 
        val_size = len(dataset) - train_size
        train_dataset, val_dataset= torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
        return train_loader, val_loader
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    return data_loader

def load_CIFAR10(root="./data/CIFAR10",isTrain=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    dataset = datasets.CIFAR10(root=root, train=isTrain, transform=transform, download=False)

    if(isTrain):
        train_size = int(0.9 * len(dataset)) 
        val_size = len(dataset) - train_size
        train_dataset, val_dataset= torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
        return train_loader, val_loader
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    return data_loader




def get_MNIST_dataset(root="./data/MNIST",isTrain=False):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    print(f"Dataset will be attempted to download to: {os.path.abspath(root)}!!!!!!!!!!!")
    
    dataset = datasets.MNIST(root=root, train=isTrain, transform=transform, download=True)

    return dataset

def get_CIFAR10_dataset(root="./data/CIFAR10",isTrain=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    print(f"Dataset will be attempted to download to: {os.path.abspath(root)}!!!!!!!!!!!")
    
    dataset = datasets.CIFAR10(root=root, train=isTrain, transform=transform, download=True)

    return dataset


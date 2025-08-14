from medmnist import PathMNIST, ChestMNIST, DermaMNIST , INFO
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def load_medmnist(dataset_name):
    info = INFO[dataset_name]
    DataClass = eval(info["python_class"])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train = DataClass(split='train', transform=transform, download=True)
    test = DataClass(split='test', transform=transform, download=True)
    
    return train, test

from medmnist import PathMNIST, ChestMNIST, DermaMNIST , INFO
from torchvision import transforms

def load_medmnist(dataset_name):
    info = INFO[dataset_name]
    DataClass = eval(info["python_class"])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train = DataClass(split='train', transform=transform, download=True)
    test = DataClass(split='test', transform=transform, download=True)
    return train, test

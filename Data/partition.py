import torch
import numpy as np
from Data.medmnis_loader import load_medmnist

def partition_noniid(config):
    train, _ = load_medmnist(config["dataset"])
    x, y = train.imgs, train.labels.flatten()
    indices = np.argsort(y)     # 🧠 Sort for non-IID (class-wise split)
    num_clients = config["num_clients"]
    chunk = len(x) // num_clients
    clients_data = []

    for i in range(num_clients):
        start, end = i * chunk, (i + 1) * chunk
        client_x = torch.tensor(x[indices[start:end]]).permute(0, 3, 1, 2).float()
        client_y = torch.tensor(y[indices[start:end]]).long()
        clients_data.append({"x": client_x, "y": client_y})

    return clients_data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def local_train(model, x, y, config):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(x, y)  # 🧪 Bundle data into PyTorch dataset
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    total_loss = 0
    num_batches = 0

    for _ in range(config["epochs"]):  # 🔁 Local training loop
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return model.state_dict(), avg_loss   # 🔁 Return updated weights only

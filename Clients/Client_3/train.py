import torch
import yaml
from Clients.train_logic import local_train
from Models.cnn import CNN
from Data.partition import partition_noniid

# Load config
with open("Config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load partitioned data (simulate 5 clients, pick index 0)
clients_data = partition_noniid(config)
data = clients_data[2]  # This is Client 1’s data

# Load a fresh global model
model = CNN(num_classes=9)

# Train locally on Client 1’s data
updated_weights = local_train(model, data["x"], data["y"], config)

# Save updated weights for server aggregation
torch.save(updated_weights, "Clients/Client_1/weights.pth")
print("✅ Client 1 training complete.")

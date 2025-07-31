from Clients.train_logic import local_train
from Models.cnn import CNN
from Server.aggregator import fed_avg
from Data.partition import partition_noniid
from Data.medmnis_loader import load_medmnist
import torch
import yaml

def evaluate(model, x_val, y_val):
    model.eval()
    correct = 0
    total = len(x_val)
    with torch.no_grad():
        for xi, yi in zip(x_val, y_val):
            # xi shape: (28, 28, 3) for PathMNIST, need to permute to (3, 28, 28)
            xi_tensor = torch.tensor(xi.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
            output = model(xi_tensor)
            pred = torch.argmax(output, dim=1).item()
            if pred == yi:
                correct += 1
    return correct / total

def federated_train(config):
    with open("Config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load PathMNIST data using your loader
    train_data, test_data = load_medmnist("pathmnist")
    x = train_data.imgs
    y = train_data.labels.flatten()
    x_val = test_data.imgs
    y_val = test_data.labels.flatten()

    config["x_val"] = x_val
    config["y_val"] = y_val

    clients_data = partition_noniid({"x": x, "y": y, **config})
    global_model = CNN(num_classes=9)

    for round in range(config["num_rounds"]):
        print(f"\n🚀 Round {round+1}")
        client_weights = []
        for i in range(config["num_clients"]):
            print(f"  🔁 Training on Client {i+1}")
            local_model = CNN(num_classes=9)
            local_model.load_state_dict(global_model.state_dict())
            weights = local_train(local_model, clients_data[i]["x"], clients_data[i]["y"], config)
            client_weights.append((weights, len(clients_data[i]["x"])))
        new_weights = fed_avg(client_weights)
        global_model.load_state_dict(new_weights)
        print("✅ Aggregated weights applied.")

        # Evaluate accuracy after each round
        acc = evaluate(global_model, x_val, y_val)
        print(f"🔍 Validation Accuracy: {acc:.4f}")
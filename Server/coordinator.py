from Clients.train_logic import local_train
from Models.cnn import CNN
from Server.aggregator import fed_avg
from Data.partition import partition_noniid
from Data.medmnis_loader import load_medmnist
import torch
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
    accuracies = []
    round_losses = []

    for round in range(config["num_rounds"]):
        print(f"\n🚀 Round {round+1}")
        client_weights = []
        round_loss = 0
        for i in range(config["num_clients"]):
            print(f"  🔁 Training on Client {i+1}")
            local_model = CNN(num_classes=9)
            local_model.load_state_dict(global_model.state_dict())
            weights, loss = local_train(local_model, clients_data[i]["x"], clients_data[i]["y"], config)
            round_loss += loss
            client_weights.append((weights, len(clients_data[i]["x"])))
        round_losses.append(round_loss / config["num_clients"])
        new_weights = fed_avg(client_weights)
        global_model.load_state_dict(new_weights)
        print("✅ Aggregated weights applied.")

        # Evaluate accuracy after each round
        acc = evaluate(global_model, x_val, y_val)
        accuracies.append(acc)
        print(f"🔍 Validation Accuracy: {acc:.4f}")

    # Plot accuracy vs. round
    plt.figure('Accuracy vs Round')
    plt.plot(range(1, config["num_rounds"] + 1), accuracies, marker='o')
    plt.xlabel("Federated Round")
    plt.ylabel("Validation Accuracy")
    plt.title("Federated Learning Accuracy Progress")
    plt.grid(True)
    plt.show()

    # After training
    plt.figure('Loss vs Round')
    plt.plot(range(1, config["num_rounds"] + 1), round_losses, marker='x')
    plt.xlabel("Federated Round")
    plt.ylabel("Average Training Loss")
    plt.title("Federated Learning Loss Progress")
    plt.grid(True)
    plt.show()

    # Get predictions for the test set
    y_pred = []
    global_model.eval()
    with torch.no_grad():
        for xi in x_val:
            xi_tensor = torch.tensor(xi.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
            output = global_model(xi_tensor)
            pred = torch.argmax(output, dim=1).item()
            y_pred.append(pred)

    # Compute confusion matrix
    cm = confusion_matrix(y_val, y_pred)

    # Plot confusion matrix
    plt.figure('confusion_matrix', figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - PathMNIST")
    plt.show()
from Clients.train_logic import local_train
from Models.cnn import CNN
from Server.aggregator import fed_avg
from Data.partition import partition_noniid
import torch
import os

def federated_train(config):
    clients_data = partition_noniid(config)      # 📦 Simulate client datasets
    global_model = CNN(num_classes=9)            # 🌍 Shared global model

    for round in range(config["num_rounds"]):
        print(f"\n🚀 Round {round+1}")
        client_weights = []

        for i in range(config["num_clients"]):
            print(f"  🔁 Training on Client {i+1}")
            local_model = CNN(num_classes=9)
            local_model.load_state_dict(global_model.state_dict())  # 📥 Copy global model
            weights = local_train(local_model, clients_data[i]["x"], clients_data[i]["y"], config)
            client_weights.append((weights, len(clients_data[i]["x"])))  # 📤 Send back weights + data size

            save_path = os.path.join("Clients", str(i+1), f"weights_round{i+1}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(weights, save_path)

        new_weights = fed_avg(client_weights)     # 🔗 Aggregate updates
        global_model.load_state_dict(new_weights) # 📥 Update global model
        print("✅ Aggregated weights applied.")

        torch.save(global_model.state_dict(), f"Server/global_model_round{i+1}.pth")
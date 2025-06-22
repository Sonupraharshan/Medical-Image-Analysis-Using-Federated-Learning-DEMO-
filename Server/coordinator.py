from Clients.train_logic import local_train
from Models.cnn import CNN
from Server.aggregator import fed_avg
from Data.partition import partition_noniid
import torch

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

        new_weights = fed_avg(client_weights)     # 🔗 Aggregate updates
        global_model.load_state_dict(new_weights) # 📥 Update global model
        print("✅ Aggregated weights applied.")

from Server.coordinator import federated_train
from Config.config_loader import load_config

if __name__ == "__main__":
    config = load_config("Config/config.yaml")   # 🔍 Load experiment settings
    federated_train(config)                      # 🚀 Start the training process

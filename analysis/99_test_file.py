import torch

# prints out the training device to check whether GPU is available
# training device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training device: {device}")
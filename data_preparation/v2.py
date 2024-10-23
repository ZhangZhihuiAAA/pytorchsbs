import torch
from torch.utils.data import TensorDataset, random_split, DataLoader


torch.manual_seed(13)

# Build tensors from numpy arrays BEFORE split
x_tensor = torch.from_numpy(x).float().reshape(-1, 1)
y_tensor = torch.from_numpy(y).float().reshape(-1, 1)

# Build dataset containing ALL data points
dataset = TensorDataset(x_tensor, y_tensor)

# Perform split
ratio = .8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train

train_data, val_data = random_split(dataset, [n_train, n_val])

# Build a loader for each set
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=16)

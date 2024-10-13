
x_train_tensor = torch.from_numpy(x_train).float().reshape(-1, 1)
y_train_tensor = torch.from_numpy(y_train).float().reshape(-1, 1)

# Build Dataset
train_data = TensorDataset(x_train_tensor, y_train_tensor)

# Build DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

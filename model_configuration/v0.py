
# This is redundant now, but it won't be when we introduce Datasets...
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set learning rate - this is eta
lr = 0.1

torch.manual_seed(42)
# Now we can create a model and send it at once to the device
model = nn.Sequential(nn.Linear(1, 1)).to(device)

# Define a SGD optimizer to update the parameters
# (now retrieved directly from the model)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Define a MSE loss function
loss_fn = nn.MSELoss(reduction='mean')

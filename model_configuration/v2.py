
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set learning rate
lr = .1

torch.manual_seed(42)

model = nn.Sequential(nn.Linear(1, 1)).to(device)

# Define an SGD optimizer to update the parameters retrieved from the model
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

loss_fn = nn.MSELoss(reduction='mean')

# Create the train_step function for model, loss function and optimizer
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)

# Create the val_step function model and loss function
val_step_fn = make_val_step_fn(model, loss_fn)

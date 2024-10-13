
n_epochs = 1000

losses = []

for epoch in range(n_epochs):
    # Perform one train step and return the corresponding loss
    loss = train_step_fn(x_train_tensor.reshape(-1, 1), y_train_tensor.reshape(-1, 1))
    losses.append(loss)


# Define number of epochs
n_epochs = 1000

for epoch in range(n_epochs):
    # Set model to TRAIN mode
    model.train()

    # Step 1 - Compute model's predictions - forward pass
    yhat = model(x_train_tensor.reshape(-1, 1))

    # Step 2 - Compute the loss
    loss = loss_fn(yhat, y_train_tensor.reshape(-1, 1))

    # Step 3 - Compute gradients for both "b" and "w" parameters
    loss.backward()

    # Step 4 - Update parameters using gradients and the learning rate
    optimizer.step()
    optimizer.zero_grad()

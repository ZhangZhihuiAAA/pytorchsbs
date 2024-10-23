import numpy as np


true_b = 1
true_w = 2
N = 100

# Data generation
rng = np.random.default_rng(54321)
x = rng.random(N)
epsilon = rng.standard_normal(N) * .1
y = true_b + true_w * x + epsilon

# Shuffle the indices
idx = np.arange(N)
np.random.shuffle(idx)

# Use the first 80 random indices for train
train_idx = idx[:int(N * .8)]
# Use the remaining indices for validation
val_idx = idx[int(N * .8):]

# Generate train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

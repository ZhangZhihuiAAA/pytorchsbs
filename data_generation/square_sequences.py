import numpy as np


def generate_sequences(n=128, variable_len=False, seed=13):
    basic_corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    rng = np.random.default_rng(seed=seed)
    bases = rng.integers(4, size=n)

    if variable_len:
        lengths = rng.integers(3, size=n) + 2
    else:
        lengths = [4] * n

    directions = rng.integers(2, size=n)
    points = [basic_corners[[(b + i) % 4 for i in range(4)]][slice(None, None, d * 2 - 1)][:l] + rng.normal(l, 2) * 0.1 
              for b, d, l in zip(bases, directions, lengths)]

    return points, directions
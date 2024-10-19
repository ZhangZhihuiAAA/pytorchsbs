import torch
from torch.utils.data import Dataset, random_split, WeightedRandomSampler


class TransformedTensorDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]

        if self.transform:
            x = self.transform(x)

        return x, self.y[index]

    def __len__(self):
        return len(self.x)


def index_splitter(n, splits, seed=13):
    '''
    Parameters:
    n: The number of data points to generate indices for.
    splits: A list of values representing the relative weights of the split sizes.
    seed: A random seed to ensure reproducibility.
    '''

    idx = torch.arange(n)

    # Make the splits argument a tensor
    splits_tensor = torch.as_tensor(splits)
    total = splits_tensor.sum().float()

    # If the total does not add up to one, divide every number by the total.
    if not total.isclose(torch.tensor(1.)):
        splits_tensor = splits_tensor / total

    # Use PyTorch random_split to split the indices
    torch.manual_seed(seed)
    return random_split(idx, splits_tensor)


def make_balanced_sampler(y):
    # Compute weights for compensating imbalanced classes
    classes, counts = y.unique(return_counts=True)
    weights = 1. / counts.float()
    sample_weights = weights[y.squeeze().long()]

    generator = torch.Generator()

    # Build the weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        generator=generator,
        replacement=True
    )

    return sampler
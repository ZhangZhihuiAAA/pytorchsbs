import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


def figure1():
    data = {'AlexNet': (61, .727, 41.8),
            'ResNet-18': (12, 2, 30.24 ),
            'ResNet-34': (22, 4, 26.7),
            'ResNet-50': (26, 4, 24.6),
            'ResNet-101': (45, 8, 23.4),
            'ResNet-152': (60, 11, 23),
            'VGG-16': (138, 16, 28.5),
            'VGG-19': (144, 20, 28.7),
            'Inception-V3': (27, 6, 22.5),
            'GoogLeNet': (13, 2, 34.2),}

    names = list(data.keys())
    stats = np.array(list(data.values()))
    xoff = [0, 0, 0, -.5, 0, 0, 0, 0, -.7, 0]
    yoff = [1.5, 0, -5., .5, 1.3, 1.5, 3.5, 3.5, .6, 0]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_title('Comparing Architectures')
    ax.set_xlabel('Number of Operations - GFLOPS')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_xlim([0, 22])
    ax.set_ylim([50, 85])
    ax.scatter(stats[:, 1], 100 - stats[:, 2], s=50*stats[:, 0], c=np.arange(12, 2, -1), cmap=plt.cm.jet)
    ax.scatter(stats[:, 1], 100 - stats[:, 2], s=4, c='w')
    for i, name in enumerate(names):
        ax.annotate(name, (stats[i, 1] - 0.65 + xoff[i], 100 - stats[i, 2] + 1.7 + yoff[i]), fontsize=12)

    fig.tight_layout()
    return fig


def compare_grayscale(converted, grayscale):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    for img, ax, title in zip([converted, grayscale], axs, ['Converted', 'Grayscale']):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.imshow(img, cmap=plt.cm.gray)
    fig.tight_layout()
    return fig


def before_batchnorm(batch):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(2):
        feature = batch[0][:, i]
        axs[i].set_title(f'mean={feature.mean():.4f} var={feature.var():.4f}')
        axs[i].set_xlabel(f'Feature #{i}')
        axs[i].set_ylabel('# of points')
        axs[i].set_ylim([0, 13])
        axs[i].hist(feature, bins=np.linspace(-3, 3, 15), alpha=0.5)
        axs[i].label_outer()
    fig.tight_layout()
    return fig


def after_batchnorm(batch, normalized):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(2):
        feature = batch[0][:, i]
        normed = normalized[:, i]
        axs[i].set_title(f'mean={normed.mean():.4f} std={normed.std(unbiased=False):.4f}')
        axs[i].set_xlabel(f'Feature #{i}')
        axs[i].set_ylabel('# of points')
        axs[i].set_ylim([0, 13])
        axs[i].hist(feature, bins=np.linspace(-3, 3, 15), alpha=0.5, label='Original')
        axs[i].hist(normed, bins=np.linspace(-3, 3, 15), alpha=0.5, label='Standardized')
        axs[i].label_outer()
        axs[i].legend()
    fig.tight_layout()
    return fig


def compare_skip(image, noskip_image, skip_image):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for img, ax, title in zip([image, noskip_image, skip_image], axs, ['Original', 'No Skip', 'Skip']):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.imshow(img, cmap=plt.cm.gray)
    fig.tight_layout()
    return fig
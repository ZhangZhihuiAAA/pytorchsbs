import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def plot_images(images, labels, n_plot=30, n_cols=16):
    n_rows = n_plot // n_cols + (n_plot % n_cols > 0)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows))
    axs = np.atleast_2d(axs)

    for i, (img, label) in enumerate(zip(images[:n_plot], labels[:n_plot])):
        row, col = i // n_cols, i % n_cols
        ax = axs[row, col]
        ax.set_title(f'#{i} - Label:{label}', size=12)
        ax.imshow(img.squeeze(), cmap='gray', vmin=0, vmax=1)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.label_outer()

    fig.tight_layout()
    return fig

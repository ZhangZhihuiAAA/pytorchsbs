import numpy as np
import matplotlib.pyplot as plt


plt.style.use('fivethirtyeight')


def figure1(x_train, y_train, x_val, y_val):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_title('Generated Data - Train')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_ylim([0, 3.1])
    axs[0].scatter(x_train, y_train)

    axs[1].set_title('Generated Data - Validation')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_ylim([0, 3.1])
    axs[1].scatter(x_val, y_val, c='r')

    fig.tight_layout()
    return fig, axs


def figure3(b, w, x_train, y_train):
    # Generate evenly spaced x features
    x_range = np.linspace(0, 1, 101)

    # Compute predictions
    yhat_range = b + w * x_range

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([0, 3.1])

    # Dataset
    ax.scatter(x_train, y_train)
    # Predictions
    ax.plot(x_range, yhat_range, label='Final model\'s predictions', c='k', linestyle='--')

    # Annotations
    ax.annotate(f'{b=:.4f}  {w=:.4f}', xy=(.4, 1.5), c='k', rotation=34)
    ax.legend(loc=0)

    fig.tight_layout()
    return fig, ax

import numpy as np

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import torch


def plot_images(images, targets, n_plot=30):
    n_rows = n_plot // 6 + ((n_plot % 6) > 0)

    fig, axs = plt.subplots(n_rows, 6, figsize=(9, 1.5 * n_rows))
    axs = np.atleast_2d(axs)

    for i, (image, target) in enumerate(zip(images[:n_plot], targets[:n_plot])):
        row, col = i // 6, i % 6
        ax = axs[row, col]
        ax.set_title(f'#{i} - Label: {target}', size=12)
        # plot filter channel in grayscale
        ax.imshow(image.squeeze(), cmap='gray', vmin=0, vmax=1)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.label_outer()

    fig.tight_layout()
    return fig


def image_channels(red, green, blue, rgb, gray, rows=(0, 1, 2)):
    fig, axs = plt.subplots(len(rows), 4, figsize=(15, 5.5))

    zeros = np.zeros((5, 5), dtype=np.uint8)

    titles0 = ['image_r', 'image_g', 'image_b', 'image_gray']
    titles1 = ['Red', 'Green', 'Blue', 'Grayscale Image']
    titles2 = ['as first channel', 'as second channel', 'as third channel', 'RGB Image']

    idx0 = np.argmax(np.array(rows) == 0)
    idx1 = np.argmax(np.array(rows) == 1)
    idx2 = np.argmax(np.array(rows) == 2)

    for i, img in enumerate([red, green, blue, gray]):
        if 0 in rows:
            axs[idx0, i].axis('off')
            axs[idx0, i].invert_yaxis()
            if (1 in rows) or (i < 3):
                axs[idx0, i].set_title(titles0[i], fontsize=16)
                axs[idx0, i].text(0.15, 0.25, str(img.astype(np.uint8)), verticalalignment='top')

        if 1 in rows:
            axs[idx1, i].set_title(titles1[i], fontsize=16)
            axs[idx1, i].set_xlabel('5x5', fontsize=14)
            axs[idx1, i].set_xticks([])
            axs[idx1, i].set_yticks([])
            axs[idx1, i].imshow(img, cmap=plt.cm.gray)
            for _, v in axs[idx1, i].spines.items():
                v.set_color('black')
                v.set_linewidth(.8)

        if 2 in rows:
            axs[idx2, i].set_title(titles2[i], fontsize=16)
            axs[idx2, i].set_xlabel(f'5x5x3 - {titles1[i][0]} only', fontsize=14)
            axs[idx2, i].set_xticks([])
            axs[idx2, i].set_yticks([])
            if i < 3:
                to_be_stacked = [zeros] * 3
                to_be_stacked[i] = img
                stacked = np.stack(to_be_stacked, axis=2)
                axs[idx2, i].imshow(stacked)
            else:
                axs[idx2, i].imshow(rgb)
            for _, v in axs[idx2, i].spines.items():
                v.set_color('black')
                v.set_linewidth(.8)

    if 1 in rows:
        axs[idx1, 0].set_ylabel('Single\nChannel\n(grayscale)', rotation=0, labelpad=40, fontsize=12)
        axs[idx1, 3].set_xlabel('5x5 = 0.21R + 0.72G + 0.07B')
    if 2 in rows:
        axs[idx2, 0].set_ylabel('Three\nChannels\n(color)', rotation=0, labelpad=40, fontsize=12)
        axs[idx2, 3].set_xlabel('5x5x3 = (R, G, B) stacked')

    fig.tight_layout()
    return fig


def figure5(sbs_logistic, sbs_nn, sbs_relu=None):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    for i in range(len(axs)):
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel('Losses')
        if sbs_relu is None:
            axs[i].set_ylim([.35, .75])
        axs[i].plot(sbs_logistic.losses if i == 0 else sbs_logistic.val_losses, 
                    'b--' if i == 0 else 'r--', 
                    label = 'Logistic - Training' if i == 0 else 'Logistic - Validation',
                    linewidth=1)
        axs[i].plot(sbs_nn.losses if i == 0 else sbs_nn.val_losses, 
                    'b' if i == 0 else 'r', 
                    label = '3-layer Network - Training' if i == 0 else '3-layer Network - Validation',
                    alpha = .5 if sbs_relu is not None else 1,
                    linewidth=1)
        if sbs_relu:
            axs[i].plot(sbs_relu.losses if i == 0 else sbs_relu.val_losses,
                        'b' if i == 0 else 'r',
                        label = 'ReLu Network - Training' if i == 0 else 'ReLu Network - Validation',
                        alpha = .8,
                        linewidth=1)
        axs[i].legend()

    fig.tight_layout()
    return fig


def weights_comparison(w_logistic_output, w_nn_equiv):
    fig = plt.figure(figsize=(15, 6))
    ax0 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((1, 3), (0, 2))

    ax0.set_title('Weights')
    ax0.set_xlabel('Parameters')
    ax0.set_ylabel('Value')
    ax0.bar(np.arange(25), w_logistic_output.cpu().numpy().squeeze(), alpha=1, label='Logistic')
    ax0.bar(np.arange(25), w_nn_equiv.cpu().numpy().squeeze(), alpha=.5, label='3-layer Network (Composed)')
    ax0.legend()

    ax1.set_title('Weights')
    ax1.set_xlabel('Logistic')
    ax1.set_ylabel('3-layer network (Composed)')
    ax1.set_xlim([-2, 2])
    ax1.set_ylim([-2, 2])
    ax1.scatter(w_logistic_output.cpu().numpy(), w_nn_equiv.cpu().numpy(), alpha=.5)

    fig.tight_layout()
    return fig


def figure7(weights):
    fig, axs = plt.subplots(1, 5, figsize=(15, 4))

    for i, w in enumerate(weights):
        axs[i].set_title(r'$w_{0' + str(i) + '}$')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].grid(False)
        axs[i].imshow(w.reshape(-1, 5).tolist(), cmap='gray')

    fig.suptitle('Hidden Layer #0')
    fig.subplots_adjust(top=.6)
    fig.tight_layout()
    return fig


def plot_activation(func, name=None):
    z = torch.linspace(-5, 5, 1000)
    z.requires_grad_(True)
    func(z).sum().backward()
    fz = func(z).detach()

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    if name is None:
        try:
            name = func.__name__
        except AttributeError:
            name = ''

    if name == 'sigmoid':
        ax.set_ylim([0, 1.1])
    elif name == 'tanh':
        ax.set_ylim([-1.1, 1.1])
    elif name == 'relu':
        ax.set_ylim([-.1, 5.01])
    else:
        ax.set_ylim([-1.1, 5.01])

    ax.set_title(name, fontsize=16)
    ax.set_xlabel('z')
    ax.set_ylabel(r'$\sigma(z)$')
    ax.set_xticks(np.arange(-5, 6))

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.plot(z.detach().numpy(), fz.numpy(), c='k', label='Activation')
    ax.plot(z.detach().numpy(), z.grad.numpy(), c='r', label='Gradient')
    ax.legend()

    fig.tight_layout()
    return fig

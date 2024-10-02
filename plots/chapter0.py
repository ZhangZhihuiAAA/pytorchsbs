import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from sklearn.linear_model import LinearRegression


plt.style.use('fivethirtyeight')


def fit_model(x_train, y_train):
    # Fits a linear regression to find the actual b and w that minimize the loss
    regression = LinearRegression()
    regression.fit(x_train.reshape(-1, 1), y_train)
    b_minimum, w_minimum = regression.intercept_, regression.coef_[0]
    return b_minimum, w_minimum


def find_index(b, w, bs, ws):
    # Looks for the closest indexes for b and w inside their respective ranges
    closest_b_idx = np.argmin(np.abs(bs[0, :] - b))
    closest_w_idx = np.argmin(np.abs(ws[:, 0] - w))

    # Closest values for b and w
    closest_b_value, closest_w_value = bs[0, closest_b_idx], ws[closest_w_idx, 0]

    return closest_b_idx, closest_w_idx, closest_b_value, closest_w_value


def figure1(x_train, y_train, x_val, y_val):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].scatter(x_train, y_train)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_ylim([0, 3.1])
    ax[0].set_title('Generated Data - Train')

    ax[1].scatter(x_val, y_val, c='r')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_ylim([0, 3.1])
    ax[1].set_title('Generated Data - Validation')

    fig.tight_layout()
    return fig, ax


def figure2(x_train, y_train, b, w, color='k'):
    # Generates evenly spaced x feature
    x_range = np.linspace(0, 1, 101)
    # Computes yhat
    yhat_range = b + w * x_range

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([0, 3])

    # Dataset
    ax.scatter(x_train, y_train)
    # Predictions
    ax.plot(x_range, yhat_range, label='Model\'s predictions', c=color, linestyle='--')

    # Annotations
    ax.annotate('b = {:.4f} w = {:.4f}'.format(b[0], w[0]), xy=(.2, .55), c=color)
    ax.legend(loc=0)

    fig.tight_layout
    return fig, ax


def figure3(x_train, y_train, b, w):
    fig, ax = figure2(x_train, y_train, b, w)
    
    # First data point
    x0, y0 = x_train[0], y_train[0]
    ax.scatter([x0], [y0], c='r')

    # Vertical line showing error between point and prediction
    ax.plot([x0, x0], [b[0] + w[0] * x0, y0 - .03], c='r', linewidth=2, linestyle='--')
    ax.arrow(x0, y0 - .03, 0, .03, color='r', shape='full', lw=0, length_includes_head=True, head_width=.03)
    ax.arrow(x0, b[0] + w[0] * x0 + .05, 0, -.03, color='r', shape='full', lw=0, length_includes_head=True, head_width=.03)
    # Annotations
    ax.annotate(r'$error_0$', xy=(.8, 1.5))

    return fig, ax


def figure4(x_train, y_train, b, w, bs, ws, all_losses):
    b_minimum, w_minimum = fit_model(x_train, y_train)

    fig = plt.figure(figsize=(12, 6))

    # 1st plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_xlabel('b')
    ax1.set_ylabel('w')
    ax1.set_title('Loss Surface')

    ax1.contour(bs[0, :], ws[:, 0], all_losses, 10, offset=-1, cmap=plt.cm.jet)

    b_idx, w_idx, _, _ = find_index(b_minimum, w_minimum, bs, ws)
    ax1.scatter(b_minimum, w_minimum, all_losses[b_idx, w_idx], c='k')
    ax1.text(-.3, 2.5, all_losses[b_idx, w_idx], 'Minimum', zdir=(1, 0, 0))
    # Random start
    b_idx, w_idx, _, _ = find_index(b, w, bs, ws)
    ax1.scatter(b, w, all_losses[b_idx, w_idx], c='k')
    # Annotations
    ax1.text(-.2, -1, all_losses[b_idx, w_idx], 'Random\n Start', zdir=(1, 0, 0))

    ax1.view_init(40, 260)

    # 2nd plot
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel('b')
    ax2.set_ylabel('w')
    ax2.set_title('Loss Surface')

    # Loss surface
    CS = ax2.contour(bs[0, :], ws[:, 0], all_losses, cmap=plt.cm.jet)
    ax2.clabel(CS, inline=1, fontsize=10)
    # Minimum
    ax2.scatter(b_minimum, w_minimum, c='k')
    # Random start
    ax2.scatter(b, w, c='k')
    # Annotations
    ax2.annotate('Random Start', xy=(.6, .05), c='k')
    ax2.annotate('Minimum', xy=(.5, 2.2), c='k')

    fig.tight_layout()
    return fig, (ax1, ax2)


# bs looks like:
# [[b1, b2, b3, ... b101],
#  [b1, b2, b3, ... b101],
#  ...
#  [b1, b2, b3, ... b101]]

# ws looks like:
# [[w1, w1, w1, ... w1],
#  [w2, w2, w2, ... w2],
#  ...
#  [w101, w101, w101, ... w101]]

# all_predictions, so all_errors and all_losses look like below
# [[b1 + w1x, b2 + w1x, b3 + w1x, ... b101 + w1x],
#  [b1 + w2x, b2 + w2x, b3 + w2x, ... b101 + w2x],
#  ...
#  [b1 + w101x, b2 + w101x, b3 + w101x, ... b101 + w101x]]


def figure5(x_train, y_train, b, w, bs, ws, all_losses):
    b_minimum, w_minimum = fit_model(x_train, y_train)

    b_idx, w_idx, fixed_b, fixed_w = find_index(b, w, bs, ws)

    w_range = ws[:, 0]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_title('Loss Surface')
    axs[0].set_xlabel('b')
    axs[0].set_ylabel('w')
    # Loss surface
    CS = axs[0].contour(bs[0, :], ws[:, 0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    # Minimum
    axs[0].scatter(b_minimum, w_minimum, c='k')
    # Starting point
    axs[0].scatter(fixed_b, fixed_w, c='k')
    # Vertical section
    axs[0].plot([fixed_b, fixed_b], w_range[[0, -1]], linestyle='--', c='r', linewidth=2)
    # Annotations
    axs[0].annotate('Minimum', xy=(.5, 2.2), c='k')
    axs[0].annotate('Random start', xy=(fixed_b + .1, fixed_w + .1), c='k')

    axs[1].set_ylim([-.1, 15.1])
    axs[1].set_xlabel('w')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Fixed: b = {:.2f}'.format(fixed_b))
    # Loss
    axs[1].plot(w_range, all_losses[:, b_idx], c='r', linestyle='--', linewidth=2)
    # Starting point
    axs[1].plot([fixed_w], [all_losses[w_idx, b_idx]], 'or')

    fig.tight_layout()
    return fig, axs


def figure6(x_train, y_train, b, w, bs, ws, all_losses):
    b_minimum, w_minimum = fit_model(x_train, y_train)

    b_idx, w_idx, fixed_b, fixed_w = find_index(b, w, bs, ws)

    b_range = bs[0, :]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_title('Loss Surface')
    axs[0].set_xlabel('b')
    axs[0].set_ylabel('w')
    # Loss surface
    CS = axs[0].contour(bs[0, :], ws[:, 0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    # Minimum
    axs[0].scatter(b_minimum, w_minimum, c='k')
    # Starting point
    axs[0].scatter(fixed_b, fixed_w, c='k')
    # Horizontal section
    axs[0].plot(b_range[[0, -1]], [fixed_w, fixed_w], linestyle='--', c='k', linewidth=2)
    # Annotations
    axs[0].annotate('Minimum', xy=(.5, 2.2), c='k')
    axs[0].annotate('Random start', xy=(fixed_b + .1, fixed_w + .1), c='k')

    axs[1].set_ylim([-.1, 15.1])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Fixed: w = {:.2f}'.format(fixed_w))
    # Loss
    axs[1].plot(b_range, all_losses[w_idx, :], c='k', linestyle='--', linewidth=2)
    # Starting point
    axs[1].plot([fixed_b], [all_losses[w_idx, b_idx]], 'ok')

    fig.tight_layout()
    return fig, axs


def figure7(b, w, bs, ws, all_losses):
    b_idx, w_idx, fixed_b, fixed_w = find_index(b, w, bs, ws)

    b_range = bs[0, :]
    w_range = ws[:, 0]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_ylim([-.1, 6.1])
    axs[0].set_xlabel('w')
    axs[0].set_ylabel('MSE (loss)')
    axs[0].set_title('Fixed: b = {:.2f}'.format(fixed_b))
    # Red rectangle
    rect = Rectangle((fixed_w - .25, all_losses[w_idx, b_idx] - .25), .5, .5)
    pc = PatchCollection([rect], facecolor='r', alpha=.3, edgecolor='r')
    axs[0].add_collection(pc)
    # Loss - fixed b
    axs[0].plot(w_range, all_losses[:, b_idx], c='r', linestyle='--', linewidth=2)
    # Starting point
    axs[0].plot([fixed_w], [all_losses[w_idx, b_idx]], 'or')

    axs[1].set_ylim([-.1, 6.1])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('MSE (loss)')
    axs[1].set_title('Fixed: w = {:.2f}'.format(fixed_w))
    axs[1].label_outer()
    # Black rectangle
    rect = Rectangle((fixed_b - .25, all_losses[w_idx, b_idx] -.25), .5, .5)
    pc = PatchCollection([rect], facecolor='k', alpha=.3, edgecolor='k')
    axs[1].add_collection(pc)
    # Loss - fixed w
    axs[1].plot(b_range, all_losses[w_idx, :], c='k', linestyle='--', linewidth=2)
    # Starting point
    axs[1].plot([fixed_b], [all_losses[w_idx, b_idx]], 'ok')

    fig.tight_layout()
    return fig, axs


def loss_curves(b_idx, w_idx, b_idx_after, w_idx_after, all_losses):
    # BEFORE
    # Loss curve for b, given w is fixed
    loss_fixed_w = all_losses[w_idx, :]
    # Loss curve for w, given b is fixed
    loss_fixed_b = all_losses[:, b_idx]
    # Loss before
    loss_before = all_losses[w_idx, b_idx]

    # AFTER
    # Loss after w is updated
    loss_after_w = all_losses[w_idx_after, b_idx]
    # Loss after b is updated
    loss_after_b = all_losses[w_idx, b_idx_after]

    return loss_fixed_b, loss_fixed_w, loss_before, loss_after_b, loss_after_w


def calc_gradient(parm_before, parm_after, loss_before, loss_after):
    # Computes changes in parm and loss
    delta_parm = parm_after - parm_before
    delta_loss = loss_after - loss_before
    # Computes gradient for parm
    manual_grad = delta_loss / delta_parm

    return manual_grad, delta_parm, delta_loss


def figure8(b, w, bs, ws, all_losses):
    b_range = bs[0, :]
    w_range = ws[:, 0]

    # BEFORE
    b_idx, w_idx, bs_before, ws_before = find_index(b, w, bs, ws)
    # AFTER
    b_idx_after, w_idx_after, bs_after, ws_after = find_index(b + .12, w + .12, bs, ws)

    loss_fixed_b, loss_fixed_w, loss_before, loss_after_b, loss_after_w = loss_curves(b_idx, w_idx, b_idx_after, w_idx_after, all_losses)

    # Computes gradient for b
    manual_grad_b, delta_b, delta_mse_b = calc_gradient(bs_before, bs_after, loss_before, loss_after_b)
    # Computes gradient for w
    manual_grad_w, delta_w, delta_mse_w = calc_gradient(ws_before, ws_after, loss_before, loss_after_w)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_ylim([.9, 1.5])
    axs[0].set_xlim([.2, .7])
    axs[0].set_xlabel('w')
    axs[0].set_ylabel('MSE (loss)')
    axs[0].set_title('Fixed: b = {:.2f}'.format(bs_before))
    # Loss curve
    axs[0].plot(w_range, loss_fixed_b, c='r', linestyle='--', linewidth=2)
    # Point - before
    axs[0].plot([ws_before], [loss_before], 'or')
    # Point - after
    axs[0].plot([ws_after], [loss_after_w], 'or')

    # Arrows
    axs[0].plot([ws_before, ws_after], [loss_before, loss_before], 'r-', linewidth=1.5)
    axs[0].arrow(ws_after, loss_before, .01, 0, color='r', shape='full', lw=0, length_includes_head=True, head_width=.01)
    axs[0].plot([ws_before, ws_before], [loss_before, loss_after_w], 'r-', linewidth=1.5)
    axs[0].arrow(ws_before, loss_after_w, 0, -.01, color='r', shape='full', lw=0, length_includes_head=True, head_width=.01)

    # Annotations
    axs[0].annotate(r'$\delta w = {:.2f}$'.format(delta_w), xy=(ws_after + .03, loss_before - .01), c='k', fontsize=15)
    axs[0].annotate(r'$\delta MSE = {:.2f}$'.format(delta_mse_w), xy=(ws_before - .08, loss_after_w - .05), c='k', fontsize=15)
    axs[0].annotate(r'$\frac{\delta MSE}{\delta w} \approx' + '{:.2f}$'.format(manual_grad_w), xy=(ws_after - .03, loss_before - .08), c='k', fontsize=15)

    axs[1].set_ylim([.9, 1.5])
    axs[1].set_xlim([.6, 1.1])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('MSE (loss)')
    axs[1].set_title('Fixed: w = {:.2f}'.format(ws_before))
    # Loss curve
    axs[1].plot(b_range, loss_fixed_w, c='k', linestyle='--', linewidth=2)
    # Point - before
    axs[1].plot([bs_before], [loss_before], 'ok')
    # Point - after
    axs[1].plot([bs_after], [loss_after_b], 'ok')

    # Arrows
    axs[1].plot([bs_before, bs_after], [loss_before, loss_before], 'k-', linewidth=1.5)
    axs[1].arrow(bs_after, loss_before, .01, 0, color='k', shape='full', lw=0, length_includes_head=True, head_width=.01)
    axs[1].plot([bs_before, bs_before], [loss_before, loss_after_b], 'k-', linewidth=1.5)
    axs[1].arrow(bs_before, loss_after_b, 0, -.01, color='k', shape='full', lw=0, length_includes_head=True, head_width=.01)

    # Annotations
    axs[1].annotate(r'$\delta b = {:.2f}$'.format(delta_b), xy=(bs_after + .03, loss_before - .01), c='k', fontsize=15)
    axs[1].annotate(r'$\delta MSE = {:.2f}$'.format(delta_mse_b), xy=(bs_before - .08, loss_after_b - .05), c='k', fontsize=15)
    axs[1].annotate(r'$\frac{\delta MSE}{\delta b} \approx' + '{:.2f}$'.format(manual_grad_b), xy=(bs_after - .05, loss_before - .1), c='k', fontsize=15)

    fig.tight_layout()
    return fig, axs


def figure9(x_train, y_train, b, w):
    # Since we updated b and w, let's regenerate the initial ones.
    # That's how using a random seed is useful, for instance.
    rng = np.random.default_rng(54321)
    b_initial = rng.standard_normal(1)
    w_initial = rng.standard_normal(1)

    fig, ax = figure2(x_train, y_train, b_initial, w_initial)

    # Generate evenly spaced x feature
    x_range = np.linspace(0, 1, 101)
    # Model's predictions for updated parameters
    yhat_range = b + w * x_range
    # Updated predictions
    ax.plot(x_range, yhat_range, label='Using parameters\nafter one update', c='g', linestyle='--')
    # Annotations
    ax.annotate('b = {:.4f} w = {:.4f}'.format(b[0], w[0]), xy=(.2, .95), c='g')
    ax.legend(loc=0)

    return fig, ax


def figure10(b, w, bs, ws, all_losses, manual_grad_b, manual_grad_w, lr):
    b_range = bs[0, :]
    w_range = ws[:, 0]

    # BEFORE
    b_idx, w_idx, bs_before, ws_before = find_index(b, w, bs, ws)
    # AFTER
    new_b_idx, new_w_idx, bs_after, ws_after = find_index(bs_before - lr * manual_grad_b,
                                                          ws_before - lr * manual_grad_w,
                                                          bs,
                                                          ws)
    # Loss before
    loss_before = all_losses[w_idx, b_idx]
    loss_fixed_b = all_losses[:, b_idx]
    loss_fixed_w = all_losses[w_idx, :]
    loss_after_b = all_losses[w_idx, new_b_idx]
    loss_after_w = all_losses[new_w_idx, b_idx]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_ylim([-.1, 6.1])
    axs[0].set_xlabel('w')
    axs[0].set_ylabel('MSE (loss)')
    axs[0].set_title('Fixed: b = {:.2f}'.format(bs_before))

    # Loss curve for w given fixed b
    axs[0].plot(w_range, loss_fixed_b, c='r', linestyle='--', linewidth=2)
    # w before update
    axs[0].plot([ws_before], [loss_before], 'or')

    # Arrows
    axs[0].arrow(ws_after, loss_before, .1, 0, color='r', shape='full', lw=0, length_includes_head=True, head_width=.1)
    axs[0].plot([ws_before, ws_after], [loss_before, loss_before], 'r-', linewidth=1.5)
    axs[0].plot([ws_after], [loss_after_w], 'or')

    # Annotations
    axs[0].annotate(r'$\eta = {:.2f}$'.format(lr), xy=(1.6, 5.5), c='k', fontsize=17)
    axs[0].annotate(r'$-\eta \frac{\delta MSE}{\delta b} \approx' + '{:.2f}$'.format(-lr * manual_grad_w), xy=(1, 2), c='k', fontsize=17)

    axs[1].set_ylim([-.1, 6.1])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('MSE (loss)')
    axs[1].set_title('Fixed: w = {:.2f}'.format(ws_before))

    # Loss curve for b given fixed w
    axs[1].plot(b_range, loss_fixed_w, c='k', linestyle='--', linewidth=2)
    # b before update
    axs[1].plot([bs_before], [loss_before], 'ok')

    # Arrows
    axs[1].arrow(bs_after, loss_before, .1, 0, color='k', shape='full', lw=0, length_includes_head=True, head_width=.1)
    axs[1].plot([bs_before, bs_after], [loss_before, loss_before], 'k-', linewidth=1.5)
    axs[1].plot([bs_after], [loss_after_b], 'ok')

    # Annotations
    axs[1].annotate(r'$\eta = {:.2f}$'.format(lr), xy=(1.6, 5.5), c='k', fontsize=17)
    axs[1].annotate(r'$-\eta \frac{\delta MSE}{\delta w} \approx' + '{:.2f}$'.format(-lr * manual_grad_b), xy=(1, 2), c='k', fontsize=17)

    fig.tight_layout()
    return fig, axs


def figure14(x_train, y_train, b, w, bad_bs, bad_ws, bad_x_train):
    bad_b_range = bad_bs[0, :]
    bad_w_range = bad_ws[:, 0]

    # So we recompute the surface for X_TRAIN using the new ranges
    all_predictions = np.apply_along_axis(
        func1d=lambda x: bad_bs + bad_ws * x,
        axis=1,
        arr=x_train.reshape(-1, 1),
    )
    all_errors = all_predictions - y_train.reshape(-1, 1, 1)
    all_losses = (all_errors ** 2).mean(axis=0)

    # Then we compute the surface for BAD_X_TRAIN using the new ranges
    bad_all_predictions = np.apply_along_axis(
        func1d=lambda x: bad_bs + bad_ws * x,
        axis=1,
        arr=bad_x_train.reshape(-1, 1),
    )
    bad_all_errors = bad_all_predictions - y_train.reshape(-1, 1, 1)
    bad_all_losses = (bad_all_errors ** 2).mean(axis=0)

    _, _, fixed_b, fixed_w = find_index(b, w, bad_bs, bad_ws)

    b_minimum, w_minimum = fit_model(x_train, y_train)

    bad_b_minimum, bad_w_minimum = fit_model(bad_x_train, y_train)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_xlabel('b')
    axs[0].set_ylabel('w')
    axs[0].set_title('Loss Surface - Before')

    # Loss surface - BEFORE
    CS = axs[0].contour(bad_bs[0, :], bad_ws[:, 0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    # Minimum point - BEFORE
    axs[0].scatter(b_minimum, w_minimum, c='k')
    # Initial random point
    axs[0].scatter(fixed_b, fixed_w, c='k')

    # Vertical cross section
    axs[0].plot([fixed_b, fixed_b], bad_w_range[[0, -1]], linestyle='--', c='r', linewidth=2)
    # Horizontal cross section
    axs[0].plot(bad_b_range[[0, -1]], [fixed_w, fixed_w], linestyle='--', c='k', linewidth=2)

    # Annotations
    axs[0].annotate('Minimum', xy=(b_minimum - .5, w_minimum + .3), c='k')
    axs[0].annotate('Random Start', xy=(fixed_b - .6, fixed_w - .3), c='k')

    axs[1].set_xlabel('b')
    axs[1].set_ylabel('w')
    axs[1].set_title('Loss Surface - After')

    # Loss surface - AFTER
    CS = axs[1].contour(bad_bs[0, :], bad_ws[:, 0], bad_all_losses, cmap=plt.cm.jet)
    axs[1].clabel(CS, inline=1, fontsize=10)
    # Minimum point - AFTER
    axs[1].scatter(bad_b_minimum, bad_w_minimum, c='k')
    # Initial random point
    axs[1].scatter(fixed_b, fixed_w, c='k')

    # Vertical cross section
    axs[1].plot([fixed_b, fixed_b], bad_w_range[[0, -1]], linestyle='--', c='r', linewidth=2)
    # Horizontal cross section
    axs[1].plot(bad_b_range[[0, -1]], [fixed_w, fixed_w], linestyle='--', c='k', linewidth=2)

    # Annotations
    axs[1].annotate('Minimum', xy=(bad_b_minimum - .5, bad_w_minimum - .3), c='k')
    axs[1].annotate('Random Start', xy=(fixed_b - .6, fixed_w + .3), c='k')

    fig.tight_layout()
    return fig, axs


def figure15(x_train, y_train, b, w, bad_bs, bad_ws, bad_x_train):
    bad_b_range = bad_bs[0, :]
    bad_w_range = bad_ws[:, 0]

    # So we recompute the surface for X_TRAIN using the new ranges
    all_predictions = np.apply_along_axis(
        func1d=lambda x: bad_bs + bad_ws * x,
        axis=1,
        arr=x_train.reshape(-1, 1),
    )
    all_errors = all_predictions - y_train.reshape(-1, 1, 1)
    all_losses = (all_errors ** 2).mean(axis=0)

    # Then we compute the surface for BAD_X_TRAIN using the new ranges
    bad_all_predictions = np.apply_along_axis(
        func1d=lambda x: bad_bs + bad_ws * x,
        axis=1,
        arr=bad_x_train.reshape(-1, 1),
    )
    bad_all_errors = bad_all_predictions - y_train.reshape(-1, 1, 1)
    bad_all_losses = (bad_all_errors ** 2).mean(axis=0)

    bad_b_idx, bad_w_idx, fixed_b, fixed_w = find_index(b, w, bad_bs, bad_ws)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_ylim([-.1, 15.1])
    axs[0].set_xlim([-1, 3.2])
    axs[0].set_xlabel('w')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Fixed: b = {:.2f}'.format(fixed_b))

    # Loss curve for b, given fixed w - BEFORE
    axs[0].plot(bad_w_range, all_losses[:, bad_b_idx], c='r', linestyle='--', linewidth=1, label='Before')
    axs[0].plot([fixed_w], [all_losses[bad_w_idx, bad_b_idx]], 'or')
    # Loss curve for b, given fixed w - AFTER
    axs[0].plot(bad_w_range, bad_all_losses[:, bad_b_idx], c='r', linestyle='--', linewidth=2, label='After')
    axs[0].plot([fixed_w], [bad_all_losses[bad_w_idx, bad_b_idx]], 'or')

    axs[0].legend()

    axs[1].set_ylim([-.1, 15.1])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Fixed: w = {:.2f}'.format(fixed_w))

    # Loss curve for w, given fixed b - BEFORE
    axs[1].plot(bad_b_range, all_losses[bad_w_idx, :], c='k', linestyle='--', linewidth=1, label='Before')
    axs[1].plot([fixed_b], [all_losses[bad_w_idx, bad_b_idx]], 'ok')
    # Loss curve for w, given fixed b - AFTER
    axs[1].plot(bad_b_range, bad_all_losses[bad_w_idx, :], c='k', linestyle='--', linewidth=2, label='After')
    axs[1].plot([fixed_b], [bad_all_losses[bad_w_idx, bad_b_idx]], 'ok')

    axs[1].legend()

    fig.tight_layout()
    return fig, axs


def figure17(x_train, y_train, scaled_bs, scaled_ws, bad_x_train, scaled_x_train):
    # So we recompute the surface for X_TRAIN using the new ranges
    all_predictions = np.apply_along_axis(
        func1d=lambda x: scaled_bs + scaled_ws * x,
        axis=1,
        arr=x_train.reshape(-1, 1),
    )
    all_errors = all_predictions - y_train.reshape(-1, 1, 1)
    all_losses = (all_errors ** 2).mean(axis=0)

    # So we recompute the surface for BAD_X_TRAIN using the new ranges
    bad_all_predictions = np.apply_along_axis(
        func1d=lambda x: scaled_bs + scaled_ws * x,
        axis=1,
        arr=bad_x_train.reshape(-1, 1),
    )
    bad_all_errors = bad_all_predictions - y_train.reshape(-1, 1, 1)
    bad_all_losses = (bad_all_errors ** 2).mean(axis=0)

    # Then we compute the surface for SCALED_X_TRAIN using the new ranges
    scaled_all_predictions = np.apply_along_axis(
        func1d=lambda x: scaled_bs + scaled_ws * x,
        axis=1,
        arr=scaled_x_train.reshape(-1, 1),
    )
    scaled_all_errors = scaled_all_predictions - y_train.reshape(-1, 1, 1)
    scaled_all_losses = (scaled_all_errors ** 2).mean(axis=0)

    b_minimum, w_minimum = fit_model(x_train, y_train)
    bad_b_minimum, bad_w_minimum = fit_model(bad_x_train, y_train)
    scaled_b_minimum, scaled_w_minimum = fit_model(scaled_x_train, y_train)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    axs[0].set_xlabel('b')
    axs[0].set_ylabel('w')
    axs[0].set_title('Loss Surface - Original')
    # Loss Surface - ORIGINAL
    CS = axs[0].contour(scaled_bs[0, :], scaled_ws[:, 0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    # Minimum point - ORIGINAL
    axs[0].scatter(b_minimum, w_minimum, c='k')
    # Annotations
    axs[0].annotate('Minimum', xy=(b_minimum - .5, w_minimum - .5), c='k')

    axs[1].set_xlabel('b')
    axs[1].set_ylabel('w')
    axs[1].set_title('Loss Surface - Bad')
    # Loss Surface - BAD
    CS = axs[1].contour(scaled_bs[0, :], scaled_ws[:, 0], bad_all_losses, cmap=plt.cm.jet)
    axs[1].clabel(CS, inline=1, fontsize=10)
    # Minimum point - BAD
    axs[1].scatter(bad_b_minimum, bad_w_minimum, c='k')
    # Annotations
    axs[1].annotate('Minimum', xy=(bad_b_minimum - .5, bad_w_minimum - .5), c='k')

    axs[2].set_xlabel('b')
    axs[2].set_ylabel('w')
    axs[2].set_title('Loss Surface - Scaled')
    # Loss Surface - BAD
    CS = axs[2].contour(scaled_bs[0, :], scaled_ws[:, 0], scaled_all_losses, cmap=plt.cm.jet)
    axs[2].clabel(CS, inline=1, fontsize=10)
    # Minimum point - BAD
    axs[2].scatter(scaled_b_minimum, scaled_w_minimum, c='k')
    # Annotations
    axs[2].annotate('Minimum', xy=(scaled_b_minimum - .5, scaled_w_minimum - .5), c='k')

    fig.tight_layout()
    return fig, axs


def figure18(x_train, y_train):
    b_minimum, w_minimum = fit_model(x_train, y_train)

    # Generates evenly spaced x feature
    x_range = np.linspace(0, 1, 101)
    # Computes yhat
    yhat_range = b_minimum + w_minimum * x_range

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([0, 3.1])

    # Dataset
    ax.scatter(x_train, y_train)
    # Predictions
    ax.plot(x_range, yhat_range, label='Final model\'s predictions', c='k', linestyle='--')

    # Annotations
    ax.annotate('b = {:.4f} w = {:.4f}'.format(b_minimum, w_minimum), xy=(b_minimum - .5, w_minimum - .5), c='k', rotation=34)
    ax.legend(loc=0)

    fig.tight_layout()
    return fig, ax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from plots.chapterextra import add_arrow, make_line


def sequence_pred(sbs_obj, X, directions=None, n_rows=2, n_cols=5):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axs = axs.flatten()

    for e, ax in enumerate(axs):
        first_corners = X[e, :2, :]
        sbs_obj.model.eval()
        next_corners = sbs_obj.model(X[e:e+1, :2].to(sbs_obj.device)).squeeze().detach().cpu().numpy()
        pred_corners = np.concatenate([first_corners, next_corners], axis=0)

        for j, corners in enumerate([X[e], pred_corners]):
            for i in range(4):
                coords = corners[i]
                color = 'k'
                ax.scatter(*coords, c=color, s=400)
                if i == 3:
                    start = -1
                else:
                    start = i
                if (not j) or (j and i):
                    ax.plot(*corners[[start, start+1]].T, c='k', lw=2, alpha=.5, linestyle='--' if j else '-')
                ax.text(*(coords - np.array([.04, 0.04])), str(i+1), c='w', fontsize=12)
                if directions is not None:
                    ax.set_title(f'{"Counter-" if not directions[e] else ""}Clockwise', fontsize=12)

        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$x_1$", rotation=0)
        ax.set_xlim([-1.5, 2.0])
        ax.set_ylim([-1.5, 2.0])

    fig.tight_layout()
    return fig


def figure9():
    english = ['the', 'European', 'economic', 'zone']
    french = ['la', 'zone',  'économique', 'européenne']

    source_labels = english
    target_labels = french

    data = np.array([[.8, 0, 0, .2],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0],
                     [0, .8, 0, .2]])

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(data, vmin=0, vmax=1, cmap=plt.cm.gray)
    ax.grid(False)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    ax.set_xticklabels(source_labels, rotation=90)

    ax.set_yticklabels([])
    ax.set_yticklabels(target_labels)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    textcolors=["white", "black"]
    kw = dict(horizontalalignment="center", verticalalignment="center")
    valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")

    threshold = im.norm(data.max())/2.
    for ip in range(data.shape[0]):
        for jp in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[ip, jp]) > threshold)])
            text = im.axes.text(jp, ip, valfmt(data[ip, jp], None), **kw)

    fig.tight_layout()
    return fig


def query_and_keys(q, ks, result=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        fig = ax.get_figure()

    norm_q = np.linalg.norm(q)
    line_q = make_line(ax, q)
    
    line_k = []
    norm_k = []
    cos_k = []
    for k in ks:
        line_k.append(make_line(ax, k))
        norm_k.append(np.linalg.norm(k))
        cos_k.append(np.dot(q, k)/(norm_k[-1]*norm_q))

    add_arrow(line_q, lw=2, color='r', text=f'||Q||={norm_q:.2f}', size=12)
    add_arrow(line_k[0], lw=2, color='k', text=r'$||K_0' + f'||={norm_k[0]:.2f}$' + '\n' + r'$cos\theta_0=' + f'{cos_k[0]:.2f}$', size=12, text_offset=(-.33, .1))
    add_arrow(line_k[1], lw=2, color='k', text=r'$||K_1' + f'||={norm_k[1]:.2f}$' + '\n' + r'$cos\theta_1=' + f'{cos_k[1]:.2f}$', size=12, text_offset=(-.63, -.18))
    add_arrow(line_k[2], lw=2, color='k', text=r'$||K_2' + f'||={norm_k[2]:.2f}$' + '\n' + r'$cos\theta_2=' + f'{cos_k[2]:.2f}$', size=12, text_offset=(.05, .58))
    if result is not None:
        add_arrow(make_line(ax, result), lw=2, color='g', text=f'Context Vector', size=12, text_offset=(-.26, .1))
    circle1 = plt.Circle((0, 0), 1., color='k', fill=False, lw=2)
    ax.add_artist(circle1)

    ax.set_ylim([-1.02, 1.02])
    ax.set_xlim([-1.02, 1.02])

    ax.set_xticks([-1.0, 0, 1.0])
    ax.set_xticklabels([-1.0, 0, 1.0], fontsize=12)
    ax.set_yticks([-1.0, 0, 1.0])
    ax.set_yticklabels([-1.0, 0, 1.0], fontsize=12)
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.set_title(r'$Query\ and\ Keys$')
    fig.tight_layout()
    return fig


def plot_attention(model, inputs, point_labels=None, source_labels=None, target_labels=None, decoder=False, self_attn=False, n_cols=5, alphas_attr='alphas'):
    textcolors=["white", "black"]
    kw = dict(horizontalalignment="center", verticalalignment="center")
    valfmt = matplotlib.ticker.StrMethodFormatter("{x:.2f}")

    model.eval()
    device = list(model.parameters())[0].device.type
    predicted_seqs = model(inputs.to(device))
    alphas = model
    for attr in alphas_attr.split('.'):
        alphas = getattr(alphas, attr)
    if len(alphas.shape) < 4:
        alphas = alphas.unsqueeze(0)
    alphas = np.array(alphas.tolist())
    n_heads, n_points, target_len, input_len = alphas.shape
    
    if point_labels is None:
        point_labels = [f'Point #{i}' for i in range(n_points)]
        
    if source_labels is None:
        source_labels = [f'Input #{i}' for i in range(input_len)]

    if target_labels is None:
        target_labels = [f'Target #{i}' for i in range(target_len)]
            
    if self_attn:
        if decoder:
            source_labels = source_labels[-1:] + target_labels[:-1]
        else:
            target_labels = source_labels        
        
    if n_heads == 1:
        n_rows = (n_points // n_cols) + int((n_points % n_cols) > 0)
    else:
        n_cols = n_heads
        n_rows = n_points

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))

    for i in range(n_points):
        for head in range(n_heads):
            data = alphas[head][i].squeeze()
            if n_heads > 1:
                if n_points > 1:
                    ax = axs[i, head]
                else:
                    ax = axs[head]
            else:
                ax = axs.flat[i]

            im = ax.imshow(data, vmin=0, vmax=1, cmap=plt.cm.gray)
            ax.grid(False)

            #ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
            #ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
            
            ax.set_xticks(np.arange(data.shape[1]))
            ax.set_yticks(np.arange(data.shape[0]))
            
            ax.set_xticklabels(source_labels)
            if n_heads == 1:
                ax.set_title(point_labels[i], fontsize=14)
            else:
                if i == 0:
                    ax.set_title(f'Attention Head #{head+1}', fontsize=14)
                if head == 0:
                    ax.set_ylabel(point_labels[i], fontsize=14)

            ax.set_yticklabels([])
            if n_heads == 1:
                if not (i % n_cols):
                    ax.set_yticklabels(target_labels)
            else:
                if head == 0:
                    ax.set_yticklabels(target_labels)            

            ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
            ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)

            ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

            threshold = im.norm(data.max())/2.
            for ip in range(data.shape[0]):
                for jp in range(data.shape[1]):
                    kw.update(color=textcolors[int(im.norm(data[ip, jp]) > threshold)])
                    text = im.axes.text(jp, ip, valfmt(data[ip, jp], None), **kw)

    fig.subplots_adjust(wspace=0.8, hspace=1.0)
    fig.tight_layout()
    return fig
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class StepByStep():
    def __init__(self, model, loss_fn, optimizer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # These attributes are defined here, but since they are
        # not available at the moment of creation, we keep them None
        self.train_loader = None
        self.val_loader = None
        self.writer = None

        # These attributes are going to be computed internally
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0
        self.visualization = {}
        self.handles = {}

        # Create the train_step function for model, loss function and optimizer
        # Note: there are NO ARGS there! It makes use of the class attributes directly
        self.train_step_fn = self._make_train_step_fn()
        # Create the val_step function for model and loss function
        self.val_step_fn = self._make_val_step_fn()

    def to(self, device):
        # This method allows the user to specify a different device
        # It sets the corresponding attribute (to be used later in
        # the mini-batches) and sends the model to the device
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass

    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_tensorboard(self, name, folder='runs'):
        # This method allows the user to create a SummaryWriter to interface with TensorBoard
        suffix = datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def train(self, n_epochs, seed=42):
        self.set_seed(seed)

        for epoch in range(n_epochs):
            # Keep track of the numbers of epochs by updating the corresponding attribute
            self.total_epochs += 1

            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            # If a SummaryWriter has been set...
            if self.writer:
                scalars = {'training': loss}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                self.writer.add_scalars(main_tag='loss',
                                        tag_scalar_dict=scalars,
                                        global_step=epoch)

        if self.writer:
            # Flush the writer
            self.writer.flush()

    def save_checkpoint(self, filename):
        checkpoint = {'epoch': self.total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'losses': self.losses,
                      'val_losses': self.val_losses}
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['losses']
        self.val_losses = checkpoint['val_losses']

        self.model.train()  # always use TRAIN for resuming training

    def predict(self, x):
        # Set it to evaluation mode for predictions
        self.model.eval()

        x_tensor = torch.as_tensor(x).float().to(self.device)
        y_hat_tensor = self.model(x_tensor)

        # Set it back to train mode
        self.model.train()

        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.plot(self.losses, label='Training Loss', c='b')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.legend()
        fig.tight_layout()
        return fig

    def add_graph(self):
        if self.train_loader and self.writer:
            # Fetche a single mini-batch so we can use add_graph
            x_sample, y_sample = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_sample.to(self.device))

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def visualize_filters(self, layer_name, **kwargs):
        try:
            # Get the layer object from the model
            layer = getattr(self.model, layer_name)

            # We are only looking at filters for 2D convolutions
            if isinstance(layer, nn.Conv2d):
                weights = layer.weight.data.cpu().numpy()
                # weights -> (ou_channels (filter), in_channels, filter_H, filter_W)
                n_filters, n_in_channels, _, _ = weights.shape

                # Build a figure
                figsize = (2 * n_in_channels + 2, 2 * n_filters)
                fig, axs = plt.subplots(n_filters, n_in_channels, figsize=figsize, squeeze=False)
                axs_array = [[axs[i, j] for j in range(n_in_channels)] for i in range(n_filters)]

                # For each filter
                for i in range(n_filters):
                    StepByStep._visualize_tensors(
                        axs_array[i],
                        weights[i],
                        layer_name=f'Filter #{i}',
                        title='Channel'
                    )

                for ax in axs.flat:
                    ax.label_outer()

            fig.tight_layout()
            return
        except AttributeError:
            return

    def attach_hooks(self, layers_to_hook, hook_fn=None):
        # Clear any previous values
        self.visualization = {}

        # Create the dictionary to map layer objects to their names
        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules[1:]}

        if hook_fn is None:
            # Hook function to be attached to the forward pass
            def hook_fn(layer, inputs, outputs):
                # Get the layer name
                name = layer_names[layer]
                # Detach the outputs
                values = outputs.detach().cpu().numpy()
                # Since the hook function may be called multiple times for example, 
                # if we make predictions for multiple mini-batches it concatenates the results.
                if self.visualization[name] is None:
                    self.visualization[name] = values
                else:
                    self.visualization[name] = np.concatenate([self.visualization[name], values])

            for name, layer in modules:
                if name in layers_to_hook:
                    # Initialize the corresponding key in the dictionary
                    self.visualization[name] = None
                    # Register the forward hook and keep the handle in another dict
                    self.handles[name] = layer.register_forward_hook(hook_fn)

    def remove_hooks(self):
        for handle in self.handles.values():
            handle.remove()
        # Clear the dict, as all hooks have been removed
        self.handles = {}

    def visualize_outputs(self, layers, n_images=16, y=None, yhat=None):
        layers = filter(lambda l: l in self.visualization.keys(), layers)
        layers = list(layers)
        shapes = [self.visualization[layer].shape for layer in layers]
        n_rows = [shape[1] if len(shape) == 4 else 1 for shape in shapes]  # number of output channels
        total_rows = np.sum(n_rows)

        fig, axs = plt.subplots(total_rows, n_images, figsize=(1.5 * n_images, 1.5 * total_rows), squeeze=False)
        axs_array = [[axs[i, j] for j in range(n_images)] for i in range(total_rows)]

        # Loop through the layers, one layer per row of subplots
        row = 0
        for i, layer in enumerate(layers):
            start_row = row
            # Take the produced feature maps for that layer
            output = self.visualization[layer]

            is_vector = len(output.shape) == 2

            for j in range(n_rows[i]):
                StepByStep._visualize_tensors(
                    axs_array[row],
                    output if is_vector else output[:, j].squeeze(),
                    y,
                    yhat,
                    layer_name=layers[i] if is_vector else f'{layers[i]}\nfil#{row - start_row}',
                    title='Image' if row == 0 else None
                )
                row += 1

        for ax in axs.flat:
            ax.label_outer()

        fig.tight_layout()
        return fig

    def correct(self, x, y, threshold=.5):
        self.model.eval()
        yhat = self.model(x.to(self.device))
        y = y.to(self.device)
        self.model.train()
        
        # We get the size of the batch and the number of classes (only 1, if it is binary)
        n_samples, n_dims = yhat.shape
        if n_dims > 1:        
            # In a multiclass classification, the biggest logit always wins, so we don't bother getting probabilities
            
            # This is PyTorch's version of argmax, but it returns a tuple: (max value, index of max value)
            _, predicted = torch.max(yhat, 1)
        else:
            n_dims += 1
            # In binary classification, we NEED to check if the last layer is a sigmoid (and then it produces probs)
            if isinstance(self.model, nn.Sequential) and isinstance(self.model[-1], nn.Sigmoid):
                predicted = (yhat > threshold).long()
            # or something else (logits), which we need to convert using a sigmoid
            else:
                predicted = (F.sigmoid(yhat) > threshold).long()
        
        # How many samples got classified correctly for each class
        result = []
        for c in range(n_dims):
            n_class = (y == c).sum().item()
            n_correct = (predicted[y == c] == c).sum().item()
            result.append((n_correct, n_class))

        return torch.tensor(result)

    @staticmethod
    def loader_apply(loader, func, reduce='sum'):
        results = [func(x, y) for i, (x, y) in enumerate(loader)]
        results = torch.stack(results, axis=0)

        if reduce == 'sum':
            results = results.sum(axis=0)
        elif reduce == 'mean':
            results = results.float().mean(axis=0)
        
        return results

    def _make_train_step_fn(self):
        # Build function that performs a step in the train loop
        def perform_train_step_fn(x, y):
            self.model.train()

            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()

        return perform_train_step_fn

    def _make_val_step_fn(self):
        # Build function that performs a step in the validation loop
        def perform_val_step_fn(x, y):
            self.model.eval()

            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)

            return loss.item()

        return perform_val_step_fn

    def _mini_batch(self, validation=False):
        # The mini-batch can be used with both loaders
        # The argument `validation` defines which loader and 
        # corresponding step function is going to be used
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn
        
        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)
        return loss

    @staticmethod
    def _visualize_tensors(axs, x, y=None, yhat=None, layer_name='', title=None):
        n_images = len(axs)
        # Gets max and min values for scaling the grayscale
        minv, maxv = np.min(x[:n_images]), np.max(x[:n_images])

        for i, img in enumerate(x[:n_images]):
            ax = axs[i]
            # Set title, labels, and remove ticks
            if title is not None:
                ax.set_title(f'{title} #{i}', fontsize=12)
            shp = np.atleast_2d(img).shape
            ax.set_ylabel(f'{layer_name}\n{shp[0]}x{shp[1]}', rotation=0, fontsize=12, labelpad=20)
            xlabel1 = '' if y is None else f'\nLabel: {y[i]}'
            xlabel2 = '' if yhat is None else f'\nPredicted: {yhat[i]}'
            xlabel = f'{xlabel1}{xlabel2}'
            if len(xlabel):
                ax.set_xlabel(xlabel, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            # Plot weight as an image
            ax.imshow(np.atleast_2d(img.squeeze()), cmap='gray', vmin=minv, vmax=maxv)

        return
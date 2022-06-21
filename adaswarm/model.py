"""Model for Iris dataset"""
import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    """Model suitable for Iris dataset"""
    def __init__(self, n_features, n_neurons, n_out):
        super(Model, self).__init__() # pylint: disable=R1725
        self.hidden = torch.nn.Linear(in_features=n_features, out_features=n_neurons)
        self.out_layer = torch.nn.Linear(in_features=n_neurons, out_features=n_out)

    def forward(self, X): # pylint: disable=C0103
        """Forward pass"""
        out = F.relu(self.hidden(X)) # pylint: disable=C0103
        out = F.sigmoid(self.out_layer(out))
        return out

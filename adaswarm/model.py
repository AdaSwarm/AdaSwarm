import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self, n_features, n_neurons, n_out):
        super(Model, self).__init__()
        self.hidden = torch.nn.Linear(in_features=n_features, out_features=n_neurons)
        self.out_layer = torch.nn.Linear(in_features=n_neurons, out_features=n_out)

    def forward(self, X):
        out = F.relu(self.hidden(X))
        out = F.sigmoid(self.out_layer(out))
        return out
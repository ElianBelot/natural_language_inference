# ===============[ IMPORTS ]===============
from .logistic_regression import max_pool
import torch.nn as nn
import torch


# ===============[ MODEL ]==========
class ShallowNeuralNetwork(nn.Module):
    """
    When called, this model does the following:
        1. Individually embed a batch of premise and hypothesis (token indices)
        2. Individually apply max_pool along the sequence length (L_p and L_h)
        3. Individually apply one feedforward layer to the pooled tensors
        4. Use ReLU on the outputs of the layer
        5. Concatenate the activated tensors into a single tensor
        6. Apply sigmoid layer to obtain predictions
    """

    # =====[ INITIALIZATION ]=====
    def __init__(self, embedding, hidden_size):
        super().__init__()

        E = embedding.weight.shape[1]
        H = hidden_size

        self.embedding = embedding
        self.ff_layer = nn.Linear(2 * E, H)
        self.activation = nn.ReLU()
        self.layer_pred = nn.Linear(H, 1)
        self.sigmoid = nn.Sigmoid()

    # =====[ FORWARD ]=====
    def forward(self, premise, hypothesis):
        # Load previously initialized layers
        emb = self.embedding
        act = self.activation
        ff_layer = self.ff_layer
        layer_pred = self.layer_pred
        sigmoid = self.sigmoid

        # Take inputs (N * L) and embed (N * L * E)
        premise = emb(premise)
        hypothesis = emb(hypothesis)

        # Max pool over L (N * E)
        premise = max_pool(premise)
        hypothesis = max_pool(hypothesis)

        # Concatenate over E (N * 2E)
        concatenated = torch.cat((premise, hypothesis), 1)

        # Go through FF layer and apply ReLU (N * H)
        output = ff_layer(concatenated)
        output = act(output)

        # Apply linear and sigmoid (N * 1)
        output = layer_pred(output)
        output = sigmoid(output)

        # Reshape to optain vector (N)
        output = torch.reshape(output, (output.shape[0],))

        return output

# ===============[ IMPORTS ]===============
from .logistic_regression import max_pool
import torch.nn as nn
import torch


# ===============[ MODEL ]==========
class DeepNeuralNetwork(nn.Module):
    """
    When called, this model does the following:
        1. Individually embed a batch of premise and hypothesis (token indices)
        2. Individually apply max_pool along the sequence length (L_p and L_h)
        3. Individually apply one feedforward layer to your pooled tensors
        4. Use the ReLU on the outputs of your layer, repeat (3) for `num_layers` times.
        5. Concatenate the activated tensors into a single tensor
        6. Apply sigmoid layer to obtain prediction
    """

    # =====[ INITIALIZATION ]=====
    def __init__(self, embedding, hidden_size, num_layers):
        super().__init__()

        E = embedding.weight.shape[1]
        H = hidden_size

        # Hidden layers
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                hidden_layers.append(nn.Linear(2 * E, H))
            else:
                hidden_layers.append(nn.Linear(H, H))

        # Other layers
        self.embedding = embedding
        self.ff_layers = nn.ModuleList(hidden_layers)
        self.activation = nn.ReLU()
        self.layer_pred = nn.Linear(H, 1)
        self.sigmoid = nn.Sigmoid()

    # =====[ FORWARD ]=====
    def forward(self, premise, hypothesis):
        # Load previously initialized layers
        emb = self.embedding
        layer_pred = self.layer_pred
        sigmoid = self.sigmoid
        ff_layers = self.ff_layers
        act = self.activation

        # Take inputs (N * L) and embed (N * L * E)
        premise = emb(premise)
        hypothesis = emb(hypothesis)

        # Max pool over L (N * E)
        premise = max_pool(premise)
        hypothesis = max_pool(hypothesis)

        # Concatenate over E (N * 2E)
        output = torch.cat((premise, hypothesis), 1)

        # Go through all FF layers and apply ReLU (N * H)
        for hidden_layer in ff_layers:
            output = hidden_layer(output)
            output = act(output)

        # Apply linear and sigmoid (N * 1)
        output = layer_pred(output)
        output = sigmoid(output)

        # Reshape to optain vector (N)
        output = torch.reshape(output, (output.shape[0],))

        return output

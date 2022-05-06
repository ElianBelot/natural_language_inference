# ===============[ IMPORTS ]===============
import torch.nn as nn
import torch


# ===============[ MAX POOLING ]===============
def max_pool(x: torch.Tensor) -> torch.Tensor:
    """
    Take the pooling over the second dimension: (N, L, D) -> (N, D).
    D is the  hidden size, N is the batch size, L is the sequence length.
    """
    return torch.max(x, 1).values


# ===============[ MODEL ]===============
class PooledLogisticRegression(nn.Module):
    """
    When called, this model does the following:
        1. Individually embed a batch of premise and hypothesis (token indices)
        2. Individually apply max_pool along the sequence length (L_p and L_h)
        3. Concatenate the pooled tensors into a single tensor
        4. Apply logistic regression to obtain predictions
    """

    # =====[ INITIALIZATION ]=====
    def __init__(self, embedding):
        super().__init__()

        self.embedding = embedding
        self.layer_pred = nn.Linear(self.embedding.weight.shape[1] * 2, 1)
        self.sigmoid = nn.Sigmoid()

    # =====[ FORWARD ]=====
    def forward(self, premise, hypothesis):

        # Load previously initialized layers
        emb = self.embedding
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

        # Apply linear and sigmoid (N * 1)
        output = layer_pred(concatenated)
        output = sigmoid(output)

        # Reshape to obtain vector (N)
        output = torch.reshape(output, (output.shape[0],))

        return output

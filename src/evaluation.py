# ===============[ IMPORTS ]===============
import torch
from . import training


# ===============[ F1 SCORE ]===============
def f1_score(y, y_pred, threshold=0.5):
    y_pred = torch.where(y_pred > 0.5, 1, 0)

    tp = (y * y_pred).sum()
    tn = ((1 - y) * (1 - y_pred)).sum()
    fp = ((1 - y) * y_pred).sum()
    fn = (y * (1 - y_pred)).sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = (2 * precision * recall) / (precision + recall)

    return f1


# ===============[ EVALUATE ]===============
def eval_run(model, loader, device='cpu'):
    """
    Iterates through a loader and predict the labels for each example.
    """

    # Prepare model and generator
    model.eval()
    generator = loader()

    # Initialize
    labels = torch.empty(0)
    predictions = torch.empty(0)

    # Iterate through batches
    for batch in generator:
        with torch.no_grad():
            y_pred = training.forward_pass(model, batch)
            y_true = torch.tensor(batch['label'])

            labels = torch.cat((labels, y_true), 0)
            predictions = torch.cat((predictions, y_pred), 0)

    return labels, predictions

# ===============[ IMPORTS ]===============
import torch

from .batching import convert_to_tensors
from .evaluation import eval_run, f1_score


# ===============[ UTILITIES ]===============
def assign_optimizer(model, **kwargs):
    return torch.optim.SGD(model.parameters(), lr=kwargs["lr"], momentum=0.9)


def bce_loss(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return -(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred)).mean()


# ===============[ FORWARD PASS ]===============
def forward_pass(model, batch, device="cpu"):
    premise = convert_to_tensors(batch["premise"])
    hypothesis = convert_to_tensors(batch["hypothesis"])

    return model(premise, hypothesis)


# ===============[ BACKWARD PASS ]===============
def backward_pass(optimizer, y, y_pred):
    # Reset gradients
    optimizer.zero_grad()

    # Automatically compute gradients
    loss = bce_loss(y, y_pred)
    loss.backward()

    # Update parameters with computed gradients
    optimizer.step()

    return loss


# ===============[ TRAINING LOOP ]===============
def train_loop(model, train_loader, valid_loader, optimizer, n_epochs=5, device="cpu"):
    # Initialize
    model.train()
    validation_scores = []
    train_batch = train_loader()

    # Train for every epoch
    for epoch in range(n_epochs):
        model.train()

        # Train for every batch
        for batch in train_batch:
            predictions = forward_pass(model, batch)
            backward_pass(optimizer, torch.tensor(batch["label"]), predictions)

            # Update parameters with computed gradients
            optimizer.step()

        # Evaluate
        y_true, y_pred = eval_run(model, valid_loader)
        score = f1_score(y_true, y_pred)
        validation_scores.append(score)

        print(f"[EPOCH {epoch}]\t{score}")

    return validation_scores

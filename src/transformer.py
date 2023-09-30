# ===============[ IMPORTS ]===============
from typing import Union

import torch
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from models.distilbert import CustomDistilBert


# ===============[ LOAD DATASETS ]===============
def load_datasets(data_directory: str) -> Union[dict, dict]:
    """
    Reads the training and validation splits from disk and load
    them into memory.
    Parameters
    ----------
    data_directory: str
        The directory where the data is stored.
    Returns
    -------
    train: dict
        The train dictionary with keys 'premise', 'hypothesis', 'label'.
    validation: dict
        The validation dictionary with keys 'premise', 'hypothesis', 'label'.
    """
    import json
    import os

    with open(os.path.join(data_directory, "train.json"), "r") as f:
        train = json.load(f)

    with open(os.path.join(data_directory, "validation.json"), "r") as f:
        valid = json.load(f)

    return train, valid


# ===============[ DATASET ]===============
class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        dd = data_dict

        if len(dd["premise"]) != len(dd["hypothesis"]) or len(dd["premise"]) != len(dd["label"]):
            raise AttributeError("Incorrect length in data_dict")

    def __len__(self):
        return len(self.data_dict["premise"])

    def __getitem__(self, idx):
        dd = self.data_dict
        return dd["premise"][idx], dd["hypothesis"][idx], dd["label"][idx]


# ===============[ TRAINING ]===============
def train_distilbert(model, loader, device):
    model.train()
    criterion = model.get_criterion()
    total_loss = 0.0

    for premise, hypothesis, target in tqdm(loader):
        optimizer.zero_grad()

        inputs = model.tokenize(premise, hypothesis).to(device)
        target = target.to(device, dtype=torch.float32)

        pred = model(inputs)

        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ===============[ EVALUATION ]===============
@torch.no_grad()
def eval_distilbert(model, loader, device):
    model.eval()

    targets = []
    preds = []

    for premise, hypothesis, target in loader:
        preds.append(model(model.tokenize(premise, hypothesis).to(device)))

        targets.append(target)

    return torch.cat(preds), torch.cat(targets)


# ===============[ TRAINING LOOP ]===============
if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(NLIDataset(train_raw), batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(NLIDataset(valid_raw), batch_size=batch_size, shuffle=False)

    # Load model and optimizer
    model = CustomDistilBert().to(device)
    optimizer = model.assign_optimizer(lr=1e-4)

    # Train
    for epoch in range(n_epochs):
        loss = train_distilbert(model, train_loader, device=device)

        preds, targets = eval_distilbert(model, valid_loader, device=device)
        preds = preds.round()

        score = f1_score(targets.cpu(), preds.cpu())
        print("Epoch:", epoch)
        print("Training loss:", loss)
        print("Validation F1 score:", score)
        print()

    train_raw, valid_raw = load_datasets(data_path)
    if sample_size is not None:
        for key in ["premise", "hypothesis", "label"]:
            train_raw[key] = train_raw[key][:sample_size]
            valid_raw[key] = valid_raw[key][:sample_size]

    full_text = train_raw["premise"] + train_raw["hypothesis"] + valid_raw["premise"] + valid_raw["hypothesis"]

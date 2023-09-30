# ===============[ IMPORTS ]===============
import random

import torch


# ===============[ BUILD LOADER ]===============
def build_loader(data_dict, batch_size=64, shuffle=False):
    """
    Builds a function that, when called, returns a generator.
    Iterate over the generator to get a batch of data (which is  a dictionary with the same keys).

    Examples
    --------
    >>> loader = build_loader(data)
    >>> for batch in loader():
            premise = batch['premise']
            label = batch['label']
    """

    def loader():
        size = len(data_dict["premise"])
        data = data_dict.items()

        if shuffle:
            order = list(range(size))
            random.shuffle(order)
            data = {key: [value[i] for i in order] for key, value in data}.items()

        return ({key: value[i : i + batch_size] for key, value in data} for i in range(0, size, batch_size))

    return loader


# ===============[ CONVERT TO TENSORS ]===============
def convert_to_tensors(text_indices):
    """
    Let N be the batch size, and L be the sequence length.
    This converts a list of lists of indices to a padded tensor of shape (N, L).
    """

    L = len(max(text_indices, key=len))
    padded = [phrase + [0] * (L - len(phrase)) for phrase in text_indices]

    return torch.tensor(padded, dtype=torch.int32)

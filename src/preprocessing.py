# ===============[ IMPORTS ]===============
import json
import os
import re


# ===============[ DATA PROCESSING ]===============
def load_datasets(data_directory):
    """
    Reads the training and validation splits from disk and loads them into memory.
    """

    with open(os.path.join(data_directory, 'train.json'), 'r') as f:
        train = json.load(f)

    with open(os.path.join(data_directory, 'validation.json'), 'r') as f:
        valid = json.load(f)

    return train, valid


def tokenize(text, max_length=None, normalize=True):
    """
    Tokenize the text into individual words (nested list of string),
    where the inner list represent a single example.

    Examples
    --------
    >>> tokenize(['Hello, world!', 'This is a test.'], normalize=True)
    [['hello', 'world'], ['this', 'is', 'a', 'test']]
    """

    if normalize:
        # Lowercase, remove non-alphanum
        regexp = re.compile('[^a-zA-Z ]+')
        text = [regexp.sub('', t.lower()) for t in text]

    return [t.split()[:max_length] for t in text]


# ===============[ WORD COUNTS ]===============
def build_word_counts(token_list):
    """
    Builds a dictionary that keeps track of how often each word appears in the dataset.
    """
    word_counts = {}

    for words in token_list:
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts


# ===============[ MAPPINGS ]===============
def build_index_map(word_counts, max_words=None):
    """
    Builds an index map that converts a word into an integer that can be accepted by THE model.
    """
    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    sorted_words = ['[PAD]'] + [item[0] for item in sorted_counts]

    return {word: ix for ix, word in enumerate(sorted_words[:max_words])}


def tokens_to_ix(tokens, index_map):
    """
    Converts a nested list of tokens to a nested list of indices using the index map.
    """
    return [[index_map[word] for word in words if word in index_map] for words in tokens]

# ===============[ IMPORTS ]===============
import argparse
import torch
import torch.nn as nn

from src.preprocessing import *
from src.training import *
from src.batching import *
from src.models.logistic_regression import PooledLogisticRegression
from src.models.shallow_network import ShallowNeuralNetwork
from src.models.deep_network import DeepNeuralNetwork


# ===============[ MAIN ]===============
if __name__ == '__main__':

    # =====[ ARGUMENTS ]=====
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='logistic', help='Model to train [logistic, shallow, deep].')
    parser.add_argument('--epochs', type=int, default=5, help='How many epochs to train the model for.')
    parser.add_argument('--device', type=str, default='cuda', required=False, help='Device to use for training [cpu, gpu].')
    parser.add_argument('--batch_size', type=int, default=64, required=False, help='Number of examples per mini-batch.')
    parser.add_argument('--embedding_dim', type=int, default=128, required=False, help='Dimension of token embeddings.')

    # =====[ INITIALIZATION ]=====
    args = parser.parse_args()
    model_name = args.model
    epochs = args.epochs
    device = args.device
    batch_size = args.batch_size
    embedding_dim = args.embedding_dim

    # =====[ LOAD DATA ]=====
    device = torch.device(device)
    train_raw, valid_raw = load_datasets('data')

    # =====[ TOKENIZE ]=====
    train_tokens = {
        'premise': tokenize(train_raw['premise'], max_length=batch_size),
        'hypothesis': tokenize(train_raw['hypothesis'], max_length=batch_size),
    }

    valid_tokens = {
        'premise': tokenize(valid_raw['premise'], max_length=batch_size),
        'hypothesis': tokenize(valid_raw['hypothesis'], max_length=batch_size),
    }

    # =====[ BUILD MAPPINGS ]=====
    word_counts = build_word_counts(
        train_tokens['premise']
        + train_tokens['hypothesis']
        + valid_tokens['premise']
        + valid_tokens['hypothesis']
    )
    index_map = build_index_map(word_counts, max_words=10_000)

    # =====[ BUILD DATASETS ]=====
    train_indices = {
        'label': train_raw['label'],
        'premise': tokens_to_ix(train_tokens['premise'], index_map),
        'hypothesis': tokens_to_ix(train_tokens['hypothesis'], index_map)
    }

    valid_indices = {
        'label': valid_raw['label'],
        'premise': tokens_to_ix(valid_tokens['premise'], index_map),
        'hypothesis': tokens_to_ix(valid_tokens['hypothesis'], index_map)
    }

    # =====[ TRAINING ]=====
    # Create loaders
    train_loader = build_loader(train_indices, batch_size=batch_size, shuffle=False)
    valid_loader = build_loader(valid_indices, batch_size=batch_size, shuffle=False)

    # This embedding layer can be trained
    embedding = nn.Embedding(10_000, embedding_dim, padding_idx=0)

    # Choose model
    if model_name == 'shallow':
        model = ShallowNeuralNetwork(embedding)
    elif model_name == 'deep':
        model = DeepNeuralNetwork(embedding)
    else:
        model = PooledLogisticRegression(embedding)

    # Assign optimizer
    optimizer = assign_optimizer(model, lr=0.003)

    # Run training loop
    train_loop(model, train_loader, valid_loader, optimizer, n_epochs=epochs)

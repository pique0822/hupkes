from Gated_GRU import GatedGRU
from datasets.dataset import Dataset

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import seaborn as sns

import pandas as pd

import argparse

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

vocabulary = []
with open('datasets/ten_tokens_explicit.txt', 'r') as vocab_file:
    for line in vocab_file:
        vocabulary.append(line.strip())

model = GatedGRU(input_size = len(vocabulary),
                 embedding_size = 2,
                 hidden_size = 15,
                 output_size = len(vocabulary))
model.load_state_dict(torch.load('models/ten_tokens_explicit_3.mdl_epoch_2990'))
model.eval()

test_batch_size = 1

for dataset_number in range(2,3):
    print('L'+str(dataset_number))

    ingestion_numbers = []

    # loading dataset
    dataset = Dataset('datasets/L'+str(dataset_number)+'/ten_tokens_explicit_singular_data.txt', 'datasets/ten_tokens_explicit.txt', 1 - 0.3, test_batch_size)

    # getting data
    dataset_hiddens = []
    dataset_updates = []
    dataset_resets  = []

    # 0 will be memory steps, 1 will be forget steps
    transitions = []

    dataset_correct = 0
    dataset_incorrect = 0

    stored_ys = {}

    text = []
    for data_idx in range(dataset.batched_testing_size()):
        X,y = dataset.get_datapoint(data_idx, training=False)

        stored_index = np.where(X[:,:dataset_number][0] == y)[0].item()

        if stored_index in stored_ys:
            stored_ys[stored_index].append(data_idx)
        else:
            stored_ys[stored_index] = []
            stored_ys[stored_index].append(data_idx)


        for line in X:
            transitions = transitions + [0]*(dataset_number+0) + [1]*(len(line)-(dataset_number+0))
            numbers_per_line = []
            line_text = []
            for token_idx, token in enumerate(line):
                if token_idx < dataset_number:
                    numbers_per_line.append(int(vocabulary[token]))

                line_text.append(dataset.token2word(token))

            ingestion_numbers.append(numbers_per_line)

            text.append(line_text)
        hidden = model.init_hidden()

        X = torch.Tensor(X).reshape(test_batch_size, -1, 1).long()
        y = torch.Tensor(y).reshape(-1).long()


        output, hidden, (update_gates, reset_gates, hidden_states) = model(X,hidden)

        line_hiddens = []
        line_updates = []
        line_resets = []
        for t in range(len(hidden_states)):
            hidden_t = hidden_states[t].reshape(-1)
            update_t = update_gates[t].reshape(-1)
            reset_t = reset_gates[t].reshape(-1)

            line_hiddens.append(hidden_t.detach().numpy())
            line_updates.append(update_t.detach().numpy())
            line_resets.append(reset_t.detach().numpy())

        dataset_hiddens.append(line_hiddens)
        dataset_updates.append(line_updates)
        dataset_resets.append(line_resets)

        class_prediction = np.argmax(output.detach().numpy())

        if class_prediction.item() == y.item():
            dataset_correct += 1
        else:
            dataset_incorrect += 1

    hiddens = np.array(dataset_hiddens)
    updates = np.array(dataset_updates)
    resets = np.array(dataset_resets)

    for line_idx, line in resets:
        plt.imshow(line.T, origin='lower')
        plt.xticks(range(line.shape[0]), line_text[line_idx])
        plt.colorbar()
        plt.show()
        import pdb; pdb.set_trace()

from Gated_GRU import GatedGRU
from datasets.dataset import Dataset

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import argparse

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser(description='Trains a GRU model on the arithmetic language dataset')
# model information
parser.add_argument('--embedding_size',type=int, default=2,
                    help='The dimension of the model embedding')
parser.add_argument('--hidden_size',type=int, default=15,
                    help='The dimension of the model hidden state')
parser.add_argument('--model_save',type=str, default=None,
                    help='File name to which we will save the model.')
args = parser.parse_args()

def apply(mode, value, symbol):
    word2val = ['zero','one','two','three','four','five','six','seven','eight','nine','ten']

    if symbol[0] == '-':
        symbol_value = -word2val.index(symbol[1:])
    else:
        symbol_value = word2val.index(symbol)

    if mode == 1: #addition
        return value + symbol_value
    elif mode == 0:
        return value - symbol_value

def cumulative_sum(expression):
    # cumulative strategy
    mode_stack = []
    # cumulative result
    result = 0
    # mode 1 is addition and mode 0 is subtraction
    mode = 1
    for symbol in expression:
        if symbol == '(':
            mode_stack.append(mode)
        elif symbol == ')':
            mode = mode_stack.pop()
        elif symbol == 'plus':
            pass
        elif symbol == 'minus':
            if mode == 0:
                mode = 1
            else:
                mode = 0
        else:
            pass
            result = apply(mode, result, symbol)
    return result


test_batch_size = 24

# for now we ignore the dataset file
L1 = Dataset('datasets/L1/data.txt', 0, test_batch_size)
L2 = Dataset('datasets/L2/data.txt', 0, test_batch_size)
L3 = Dataset('datasets/L3/data.txt', 0, test_batch_size)
L4 = Dataset('datasets/L4/data.txt', 0, test_batch_size)
L5 = Dataset('datasets/L5/data.txt', 0, test_batch_size)
L6 = Dataset('datasets/L6/data.txt', 0, test_batch_size)
L7 = Dataset('datasets/L7/data.txt', 0, test_batch_size)
L8 = Dataset('datasets/L6/data.txt', 0, test_batch_size)
L9 = Dataset('datasets/L7/data.txt', 0, test_batch_size)

vocabulary = ['zero', 'one', 'two', 'three', 'four', 'five',
              'six', 'seven', 'eight', 'nine', 'ten',
              '-one',  '-two', '-three', '-four', '-five',
              '-six', '-seven', '-eight', '-nine', '-ten',
              '(', ')', 'plus', 'minus']

L1.set_vocabulary(vocabulary)
L2.set_vocabulary(vocabulary)
L3.set_vocabulary(vocabulary)
L4.set_vocabulary(vocabulary)
L5.set_vocabulary(vocabulary)
L6.set_vocabulary(vocabulary)
L7.set_vocabulary(vocabulary)
L8.set_vocabulary(vocabulary)

datasets = [L1,L2,L3,L4,L5,L6,L7,L8,L9]

model = GatedGRU(input_size = len(vocabulary),
                 embedding_size = args.embedding_size,
                 hidden_size = args.hidden_size,
                 output_size = 1)
model.load_state_dict(torch.load(args.model_save))
model.eval()

criterion = nn.MSELoss()

# MSE over all datasets
all_MSEs = []

cumulative_sum_dataset = {}
dataset_hiddens = {}
for dataset_number, dataset in enumerate(datasets):
    cumulative_sum_dataset[dataset_number] = []
    dataset_hiddens[dataset_number] = []

    print('L'+str(dataset_number+1))
    dataset_MSE = 0
    for data_idx in range(dataset.batched_testing_size()):
        X,y = dataset.get_datapoint(data_idx, training=False)

        for line in X:
            cumulative_sum_results = []
            words = []
            for token in line:
                words.append(dataset.token2word(token))
                result = cumulative_sum(words)
                cumulative_sum_results.append(result)

            cumulative_sum_dataset[dataset_number].append(cumulative_sum_results)

        X = torch.Tensor(X).reshape(X.shape[0], -1, 1).long()
        y = torch.Tensor(y).reshape(X.shape[0],-1)


        hidden = model.init_hidden()

        output, hidden, (update_gates, reset_gates, hidden_states) = model(X,hidden)

        import pdb; pdb.set_trace()

        dataset_MSE += criterion(output,y)*X.shape[0]

    all_MSEs.append(dataset_MSE/dataset.batched_testing_size())

print('Plotting')
plt.plot(range(len(datasets)),all_MSEs)
plt.xticks(range(len(datasets)),['L1','L2','L3','L4','L5','L6','L7','L8','L9'])
plt.xlabel('Dataset')
plt.ylabel('MSE')
plt.show()

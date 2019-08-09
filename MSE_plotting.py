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

# for now we ignore the dataset file
L1 = Dataset('datasets/L1/data.txt', 0, 24)
L2 = Dataset('datasets/L2/data.txt', 0, 24)
L3 = Dataset('datasets/L3/data.txt', 0, 24)
L4 = Dataset('datasets/L4/data.txt', 0, 24)
L5 = Dataset('datasets/L5/data.txt', 0, 24)
L6 = Dataset('datasets/L6/data.txt', 0, 24)
L7 = Dataset('datasets/L7/data.txt', 0, 24)
L8 = Dataset('datasets/L6/data.txt', 0, 24)
L9 = Dataset('datasets/L7/data.txt', 0, 24)

datasets = [L1,L2,L3,L4,L5,L6,L7,L8,L9]

model = GatedGRU(input_size = len(L9.get_vocabulary()),
                 embedding_size = args.embedding_size,
                 hidden_size = args.hidden_size,
                 output_size = 1)
model.load_state_dict(torch.load(args.model_save))
model.eval()

all_MSEs = []
for dataset in datasets:
    for data_idx in range(dataset.batched_testing_size()):
        X,y = dataset.get_datapoint(data_idx, training=False)

        X = torch.Tensor(X).reshape(24, -1, 1).long()
        y = torch.Tensor(y).reshape(24,-1)

        import pdb; pdb.set_trace()

        hidden = model.init_hidden()

        output, hidden, _ = model(X,hidden)

        MSE = criterion(output,y)

        all_MSEs.append(MSE)

plt.plot(range(len(datasets)),all_MSEs)
plt.xticks(['L1','L2','L3','L4','L5','L6','L7','L8','L9'])

plt.show()

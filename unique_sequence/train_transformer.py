from Gated_GRU import GatedGRU
from datasets.sequence_generator import generate_examples

import argparse

import torch.nn as nn
import torch

import harvard_transformer as tr
from torch.autograd import Variable


import numpy as np
import os
if not os.path.exists('models'):
    os.makedirs('models')
# python3 train_transformer.py --dataset_file ten_tokens_explicit_singular_data.txt --vocabulary_file datasets/ten_tokens_explicit.txt --model_save test.mdl
parser = argparse.ArgumentParser(description='Trains a GRU model on the arithmetic language dataset')

# dataset information
parser.add_argument('--dataset_file',type=str, default="ten_tokens_explicit_singular_data.txt",
                    help='File that contains the full dataset from which we will take the training and test data.')
parser.add_argument('--training_percent',type=float,
                    default=0.6,
                    help='Percent of data that is to be used as training.')
parser.add_argument('--batch_size',type=int,
                    default=24,
                    help='Batch size for the dataset')
parser.add_argument('--num_batches',type=int,
                    default=100,
                    help='Batch size for the dataset')
parser.add_argument('--vocabulary_file',type=str,
                    default='datasets/ten_tokens_explicit.txt',
                    help='File containing each possible word in the training set.')
parser.add_argument('--dataset_seed',type=int,
                    default=None,
                    help='Randomization seed for dataset.')
parser.add_argument('--operation_type',type=str,
                    default='singular',
                    help='{singular | combined}')
parser.add_argument('--transition_type',type=str,
                    default='explicit',
                    help='{implicit | explicit | repeated}')


# model information
parser.add_argument('--embedding_size',type=int, default=2,
                    help='The dimension of the model embedding')
parser.add_argument('--hidden_size',type=int, default=15,
                    help='The dimension of the model hidden state')
parser.add_argument('--model_save',type=str, default=None,
                    help='File name to which we will save the model.')

# hyper parameters
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Number of epochs.')
parser.add_argument('--print_frequency', type=int, default=10,
                    help='The distance between printing loss information.')

args = parser.parse_args()

vocabulary = []
with open(args.vocabulary_file) as file:
    for line in file:
        vocabulary.append(line.strip())

def data_generator(vocabulary, batch_size, num_batches, Ks=[2,4,5,7]):
    "Generate random data for a src-tgt copy task."
    for btch in range(num_batches):
        selected_ks = np.random.choice(Ks, batch_size)

        pad = len(vocabulary)
        srcs = []
        tgts = []
        for lk in selected_ks:
            output = generate_examples(transition_type = args.transition_type, \
                                        operation_type = args.operation_type, \
                                        vocabulary = vocabulary, \
                                        k = lk, \
                                        num_examples=1).strip()

            training_line, target = output.split(';')
            training_sequence = training_line.split(' ')

            src = []
            for char in training_sequence:
                src.append(vocabulary.index(char))

            for padding_char in range(20 - len(training_sequence)):
                src.append(pad)

            tgt = [vocabulary.index(target)]
            for padding_char in range(20 - len(tgt)):
                tgt.append(pad)

            srcs.append(src)
            tgts.append(tgt)

        srcs = torch.from_numpy(np.array(srcs))
        tgts = torch.from_numpy(np.array(tgts))

        # data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        # data[:, 0] = 1
        srcs = Variable(srcs, requires_grad=False)
        tgts = Variable(tgts, requires_grad=False)
        # import pdb; pdb.set_trace()

        yield tr.Batch(srcs, tgts, pad)

# import pdb; pdb.set_trace()
model = tr.make_transformer(src_vocab=len(vocabulary)+1, \
                        tgt_vocab=len(vocabulary)+1, \
                        N=1, \
                        d_model=16, d_ff=64, h=2, dropout=0.0)

criterion = nn.CrossEntropyLoss()

model_opt = tr.NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
for epoch in range(args.num_epochs):
    model.train()
    tr.run_epoch(data_generator(vocabulary, args.batch_size, args.num_batches), \
                model, \
              tr.SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(tr.run_epoch(data_generator(vocabulary, args.batch_size, 1), \
                model, \
              tr.SimpleLossCompute(model.generator, criterion, None)))

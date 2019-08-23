from Gated_GRU import GatedGRU
from datasets.dataset import Dataset

import argparse

import torch.nn as nn
import torch

import numpy as np
import os
if not os.path.exists('models'):
    os.makedirs('models')
# python3 train_GRU.py --dataset_file ten_tokens_singular_data.txt --vocabulary_file datasets/ten_tokens.txt --model_save test.mdl
parser = argparse.ArgumentParser(description='Trains a GRU model on the arithmetic language dataset')

# dataset information
parser.add_argument('--dataset_file',type=str, default=None,
                    help='File that contains the full dataset from which we will take the training and test data.')
parser.add_argument('--training_percent',type=float,
                    default=0.6,
                    help='Percent of data that is to be used as training.')
parser.add_argument('--batch_size',type=int,
                    default=24,
                    help='Batch size for the dataset')
parser.add_argument('--vocabulary_file',type=str,
                    default='datasets/ten_tokens.txt',
                    help='File containing each possible word in the training set.')
parser.add_argument('--dataset_seed',type=int,
                    default=None,
                    help='Randomization seed for dataset.')

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

# optimizer parameters
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--b1', type=float, default=0.9,
                    help='A coefficient used for computing running averages')
parser.add_argument('--b2', type=float, default=0.999,
                    help='A coefficient used for computing running averages')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='Term added to improve numerical stability')
parser.add_argument('--decay', type=float, default=0.0,
                    help='weight decay')

args = parser.parse_args()


L2 = Dataset(dataset_file='datasets/L2/'+args.dataset_file, \
             vocabulary_file=args.vocabulary_file, \
             training_percent=args.training_percent,\
             batch_size=args.batch_size,\
             seed=args.seed)
L4 = Dataset(dataset_file='datasets/L4/'+args.dataset_file, \
             vocabulary_file=args.vocabulary_file, \
             training_percent=args.training_percent,\
             batch_size=args.batch_size,\
             seed=args.seed)
L5 = Dataset(dataset_file='datasets/L5/'+args.dataset_file, \
             vocabulary_file=args.vocabulary_file, \
             training_percent=args.training_percent,\
             batch_size=args.batch_size,\
             seed=args.seed)
L7 = Dataset(dataset_file='datasets/L7/'+args.dataset_file, \
             vocabulary_file=args.vocabulary_file, \
             training_percent=args.training_percent,\
             batch_size=args.batch_size,\
             seed=args.seed)

vocabulary = L7.get_vocabulary()

datasets = [L2,L4,L5,L7]

# defining the training sequence
sequence = [0]*L2.batched_training_size() + \
           [1]*L4.batched_training_size() + \
           [2]*L5.batched_training_size() + \
           [3]*L7.batched_training_size()

model = GatedGRU(input_size = len(vocabulary),
                 embedding_size = args.embedding_size,
                 hidden_size = args.hidden_size,
                 output_size = len(vocabulary))

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1,args.b2), eps=args.eps, weight_decay=args.decay)

for epoch in range(args.num_epochs):
    epoch_loss = 0

    training_sequence = np.random.choice(sequence,len(sequence),replace=False)

    dataset_indices = [0]*len(datasets)

    for training_sequence_idx in training_sequence:

        dataset_index = dataset_indices[training_sequence_idx]

        optimizer.zero_grad()

        hidden = model.init_hidden()

        X,y = datasets[training_sequence_idx].get_datapoint(dataset_index, training=True)
        X = torch.Tensor(X).reshape(args.batch_size, -1, 1).long()
        y = torch.Tensor(y).reshape(-1).long()

        output, hidden, _ = model(X,hidden)
        output = output

        batch_loss = criterion(output,y)

        epoch_loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()

        dataset_indices[training_sequence_idx] = dataset_index + 1

    if epoch%args.print_frequency == 0:
        print('Epoch '+str(epoch)+' - Loss: ',round(epoch_loss,2))
        iterim_model = model
        torch.save(iterim_model.state_dict(), args.model_save+'_epoch_'+str(epoch))

# final test
epoch_loss = 0

training_sequence = np.random.choice(sequence,len(sequence),replace=False)

dataset_indices = [0]*len(datasets)

for training_sequence_idx in training_sequence:

    dataset_index = dataset_indices[training_sequence_idx]

    optimizer.zero_grad()

    hidden = model.init_hidden()

    X,y = datasets[training_sequence_idx].get_datapoint(dataset_index, training=True)
    X = torch.Tensor(X).reshape(args.batch_size, -1, 1).long()
    y = torch.Tensor(y).reshape(-1).long()

    output, hidden, _ = model(X,hidden)
    output = output

    batch_loss = criterion(output,y)

    epoch_loss += batch_loss.item()
    batch_loss.backward()
    optimizer.step()

    dataset_indices[training_sequence_idx] = dataset_index + 1
print('Epoch '+str(args.num_epochs)+' - Loss: ',round(epoch_loss,2))

# save model
model = model
torch.save(model.state_dict(), args.model_save)

#14344192

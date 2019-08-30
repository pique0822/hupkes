from Gated_GRU import GatedGRU
from datasets.sequence_generator import generate_examples

import argparse

import torch.nn as nn
import torch

import harvard_transformer as tr
from torch.autograd import Variable


import numpy as np
import os
parser = argparse.ArgumentParser(description='Trains a GRU model on the arithmetic language dataset')

# dataset information
parser.add_argument('--dataset_file',type=str, default="ten_tokens_explicit_singular_data.txt",
                    help='File that contains the full dataset from which we will take the training and test data.')
parser.add_argument('--training_percent',type=float,
                    default=0.6,
                    help='Percent of data that is to be used as training.')
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
parser.add_argument('--num_layers',type=int, default=2,
                    help='The dimension of the model layers')
parser.add_argument('--num_heads',type=int, default=2,
                    help='The dimension of the model heads')
parser.add_argument('--hidden_size',type=int, default=16,
                    help='The dimension of the model hidden state')
parser.add_argument('--model_save',type=str, default=None,
                    help='File name to which we will save the model.')
args = parser.parse_args()
# hyper parameters
vocabulary = []
with open(args.vocabulary_file) as file:
    for line in file:
        vocabulary.append(line.strip())

def data_sample(vocabulary, Ks=[2,4,5,7]):
    "Generate random data for a src-tgt copy task."
    pad = len(vocabulary)
    output = generate_examples(transition_type = args.transition_type, \
                               operation_type = args.operation_type, \
                               vocabulary = vocabulary, \
                               k = np.random.choice(Ks), \
                               num_examples=1).strip()

    training_line, target = output.split(';')
    training_sequence = training_line.split(' ')

    src = []
    for char in training_sequence:
        src.append(vocabulary.index(char))

    for padding_char in range(20 - len(training_sequence)):
        src.append(pad)

    tgt = [pad, vocabulary.index(target)]
            # for padding_char in range(20 - len(tgt)):
            #     tgt.append(pad)

    src = Variable(torch.from_numpy(np.array(src)), requires_grad=False)
    tgt = Variable(torch.from_numpy(np.array(tgt)), requires_grad=False)
        # import pdb; pdb.set_trace()
    return src, tgt

model = tr.make_transformer(src_vocab=len(vocabulary)+1, \
                        tgt_vocab=len(vocabulary)+1, \
                        N=args.num_layers, \
                        d_model=args.hidden_size, \
                        d_ff=4*args.hidden_size, \
                        h=args.num_heads, dropout=0.0)
model.load_state_dict(torch.load(args.model_save))
model.eval()

for k in range(2,11):
	correct = 0
	for ex in range(2000):
		

		src, tgt = data_sample(vocabulary, Ks=[k])
		src = src.reshape(1,-1)
		true = tgt[1].item()

		src_mask = (src != len(vocabulary)).unsqueeze(-2)
		out = tr.greedy_decode(model, src, src_mask, max_len=2, start_symbol=len(vocabulary))

		pred = out[0][1].item()

		# print(pred, true)
		if pred == true:
			# print('CORR')
			correct += 1
	# import pdb; pdb.set_trace()
	print('Dataset L'+str(k)+' Accuracy: '+str(round(correct/2000*100, 2)) + '%')


# import pdb; pdb.set_trace()


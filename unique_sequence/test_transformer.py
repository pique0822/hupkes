from Gated_GRU import GatedGRU
from datasets.sequence_generator import generate_examples

import argparse

import torch.nn as nn
import torch

import harvard_transformer as tr
from torch.autograd import Variable


import numpy as np
import os

vocabulary = []
with open('datasets/ten_tokens_explicit.txt') as file:
    for line in file:
        vocabulary.append(line.strip())

def data_sample(vocabulary, Ks=[2,4,5,7]):
    "Generate random data for a src-tgt copy task."
    pad = len(vocabulary)
    output = generate_examples(transition_type = 'explicit', \
                               operation_type = 'singular', \
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
                        N=2, \
                        d_model=16, d_ff=64, h=2, dropout=0.0)
model.load_state_dict(torch.load('test.mdl'))
model.eval()

for k in range(2,11):
	for ex in range(2000):
		correct = 0

		src, tgt = data_sample(vocabulary, Ks=[k])
		src = src.reshape(1,-1)
		true = tgt[1]

		src_mask = (src != len(vocabulary)).unsqueeze(-2)
		out = tr.greedy_decode(model, src, src_mask, max_len=2, start_symbol=len(vocabulary))

		pred = out[0][1].item()

		if pred == true:
			correct += 1

	print('Dataset L'+str(k)+' Accuracy: '+str(round(correct/2000*100, 2)) + '%')


# import pdb; pdb.set_trace()


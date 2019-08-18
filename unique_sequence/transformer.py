# General outline
# Get token sequence
# embed tokens
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.modules.normalization as norm
from torch.nn.parameter import Parameter
from enum import IntEnum

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class Attention(nn.Module):
    """Tokens must be passed into the network"""
    def __init__(self, input_size, embedding_size, prior_only=False):
        super().__init__()

        self.prior_only = prior_only

        self.input_size = input_size
        self.embedding_size = embedding_size

        self.wq = Parameter(torch.Tensor(input_size, self.embedding_size))
        self.wk = Parameter(torch.Tensor(input_size, self.embedding_size))
       	self.wv = Parameter(torch.Tensor(input_size, self.embedding_size))

        self.init_weights()

    def init_weights(self):
    	for p in self.parameters():
        	if p.data.ndimension() >= 2:
        		nn.init.xavier_uniform_(p.data)
        	else:
        		nn.init.zeros_(p.data)

    def forward(self, x):
    	# the expected shape of x is (batch, sequence, feature)

        dk = math.sqrt(self.wk.shape[1])
        
        x_q = x @ self.wq
        x_v = x @ self.wv
        x_k = x @ self.wk

        queried_keys = x_q @ x_k.t()

        if self.prior_only:
        	upper_tri_indices = np.triu_indices(queried_keys.shape[0], k=1)

        	inf_replace = torch.zeros(queried_keys.shape[0],queried_keys.shape[0])
        	inf_replace[upper_tri_indices] = -np.inf

        	queried_keys = queried_keys + inf_replace

        scores = nn.functional.softmax(queried_keys/dk)
        z = scores @ x_v
        
        return z, (scores, x_q, x_v, x_k)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, attention_heads = 1):
        super().__init__()

        if attention_heads < 1:
        	print('attention_heads must be 1 or greater')
        	raise ValueError

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        heads = []
        for head in range(attention_heads):
        	heads.append(Attention(self.input_size, self.embedding_size))

        self.attention_heads = nn.ModuleList(heads)

        self.attention_size = self.input_size

        self.multiheaded_transformer = Parameter(torch.Tensor(self.embedding_size * attention_heads, self.attention_size))

        self.attention_normalize = norm.LayerNorm(self.attention_size)

        self.ffn = nn.Linear(self.attention_size, self.hidden_size)

        self.ffn_normalize = norm.LayerNorm(self.hidden_size)

        self.init_weights()

    def init_weights(self):
    	for p in self.parameters():
        	if p.data.ndimension() >= 2:
        		nn.init.xavier_uniform_(p.data)
        	else:
        		nn.init.zeros_(p.data)

    def forward(self, x):
    	# the expected shape of x is (batch, sequence, feature)

        attention_z = []

        attention_scores = []
        attention_xqs = []
        attention_xvs = []
        attention_xks = []


        for attention in self.attention_heads:
        	z, (scores, x_q, x_v, x_k) = attention(x)

        	attention_z.append(z)
        	attention_scores.append(scores)
        	attention_xqs.append(x_q)
        	attention_xvs.append(x_v)
        	attention_xks.append(x_k)
        
        concatenated_z = torch.cat(attention_z, 1)

        transformed_z = concatenated_z @ self.multiheaded_transformer 

        normed_z = self.attention_normalize(transformed_z + x)

        output = self.ffn(normed_z)

        normed_output = self.ffn_normalize(output + normed_z)
        
        return normed_output, (attention_z, attention_scores, attention_xqs, attention_xvs, attention_xks)

class LayeredEncoder(nn.Module):
    """Tokens must be passed into the network"""
    def __init__(self, input_size, embedding_sizes, hidden_sizes, attention_heads = None, attention_embedding = None):
        super().__init__()
        self.input_size = input_size
        self.embedding_sizes = embedding_sizes
        self.hidden_sizes = hidden_sizes

        if attention_heads is None:
        	attention_heads = [1]*len(hidden_sizes)

        if attention_embedding is None:
        	attention_embedding = [None]*len(hidden_sizes)

        self.layers = [Encoder(input_size, embedding_sizes[0], hidden_sizes[0], attention_heads[0], attention_embedding[0])]

        for layer in range(1, len(embedding_sizes)):
        	self.layers.append(Encoder(hidden_sizes[layer - 1], embedding_sizes[layer], hidden_sizes[layer], attention_heads[layer], attention_embedding[layer]))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
    	# the expected shape of x is (batch, sequence, feature)
    	all_zs = []
    	all_scores = []
    	all_xqs = []
    	all_xvs = []
    	all_xks = []
    	all_zs = []
    	all_outputs = []

    	output = x
    	for layer in range(len(self.layers)):
    		output, (z, scores, x_q, x_v, x_k) = self.layers[layer](output)

    		all_outputs.append(output)
    		all_zs.append(z)
    		all_scores.append(scores)
    		all_xqs.append(x_q)
    		all_xvs.append(x_v)
    		all_xks.append(x_k)

    	return output, (all_outputs, all_zs, all_scores, all_xqs, all_xvs, all_xks)

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, attention_heads = 1):
        super().__init__()

        if attention_heads < 1:
        	print('attention_heads must be 1 or greater')
        	raise ValueError

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        heads = []
        for head in range(attention_heads):
        	heads.append(Attention(self.input_size, self.embedding_size))

        self.attention_heads = nn.ModuleList(heads)

        self.attention_size = self.input_size

        self.multiheaded_transformer = Parameter(torch.Tensor(self.embedding_size * attention_heads, self.attention_size))

        self.attention_normalize = norm.LayerNorm(self.attention_size)

        self.ffn = nn.Linear(self.attention_size, self.hidden_size)

        self.ffn_normalize = norm.LayerNorm(self.hidden_size)

        self.init_weights()

    def init_weights(self):
    	for p in self.parameters():
        	if p.data.ndimension() >= 2:
        		nn.init.xavier_uniform_(p.data)
        	else:
        		nn.init.zeros_(p.data)

    def forward(self, x):
    	# the expected shape of x is (batch, sequence, feature)

        attention_z = []

        attention_scores = []
        attention_xqs = []
        attention_xvs = []
        attention_xks = []


        for attention in self.attention_heads:
        	z, (scores, x_q, x_v, x_k) = attention(x)

        	attention_z.append(z)
        	attention_scores.append(scores)
        	attention_xqs.append(x_q)
        	attention_xvs.append(x_v)
        	attention_xks.append(x_k)
        
        concatenated_z = torch.cat(attention_z, 1)

        transformed_z = concatenated_z @ self.multiheaded_transformer 

        normed_z = self.attention_normalize(transformed_z + x)

        output = self.ffn(normed_z)

        normed_output = self.ffn_normalize(output + normed_z)
        
        return normed_output, (attention_z, attention_scores, attention_xqs, attention_xvs, attention_xks)


if __name__ == '__main__':
	x = torch.rand(5,10)

	
	a = Attention(input_size=10, embedding_size=20, prior_only=True)

	e = Encoder(input_size = 10, embedding_size=20, hidden_size=30, attention_heads = 2)

	a(x)

	output, (z, scores, x_q, x_v, x_k) = e.forward(x)

	el = LayeredEncoder(input_size = 10, embedding_sizes=[20,30,40], hidden_sizes=[50,60,70])
	
	output, _ = el.forward(x)
	import pdb; pdb.set_trace()








#EOF
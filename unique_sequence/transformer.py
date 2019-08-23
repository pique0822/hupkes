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

class PointwiseFFN(nn.Module):
    """Tokens must be passed into the network"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.output_size)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, encoded = None):
        # the expected shape of x is (batch, sequence, feature)
        return self.l2(nn.functional.relu(self.l1(x)))

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

    def forward(self, x, encoded = None):
    	# the expected shape of x is (batch, sequence, feature)

        dk = math.sqrt(self.wk.shape[1])

        if encoded is None:
            Q = x @ self.wq
            V = x @ self.wv
            K = x @ self.wk
        else:
            Q = x @ self.wq
            V = encoded @ self.wv
            K = encoded @ self.wk


        queried_keys = Q @ K.t()

        if self.prior_only:
        	upper_tri_indices = np.triu_indices(queried_keys.shape[0], k=1)

        	inf_replace = torch.zeros(queried_keys.shape[0],queried_keys.shape[0])
        	inf_replace[upper_tri_indices] = -np.inf

        	queried_keys = queried_keys + inf_replace

        scores = nn.functional.softmax(queried_keys/dk)
        z = scores @ V
        
        return z, (scores, Q, V, K)

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

        self.multiheaded_selfattention = Parameter(torch.Tensor(self.embedding_size * attention_heads, self.input_size))

        self.attention_normalize = norm.LayerNorm(self.input_size)

        self.ffn = PointwiseFFN(self.input_size, self.hidden_size, self.input_size)

        self.ffn_normalize = norm.LayerNorm(self.input_size)

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

        transformed_z = concatenated_z @ self.multiheaded_selfattention 

        normed_z = self.attention_normalize(transformed_z + x)

        output = self.ffn(normed_z)

        normed_output = self.ffn_normalize(output + normed_z)
        
        return normed_output, (attention_z, attention_scores, attention_xqs, attention_xvs, attention_xks)

class LayeredEncoder(nn.Module):
    """Tokens must be passed into the network"""
    def __init__(self, input_size, embedding_sizes, hidden_sizes, attention_heads = None):
        super().__init__()
        self.input_size = input_size
        self.embedding_sizes = embedding_sizes
        self.hidden_sizes = hidden_sizes

        if attention_heads is None:
        	self.attention_heads = [1]*len(embedding_sizes)
        elif type(attention_heads) is int:
            self.attention_heads = [attention_heads]*len(embedding_sizes)
        else:
            self.attention_heads = attention_heads

        self.layers = [Encoder(self.input_size, self.embedding_sizes[0], self.hidden_sizes[0], self.attention_heads[0])]

        for layer in range(1, len(self.embedding_sizes)):
        	self.layers.append(Encoder(self.input_size, self.embedding_sizes[layer], self.hidden_sizes[layer], self.attention_heads[layer]))

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


# Decoder
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
        encdec_heads = []

        for head in range(attention_heads):
            heads.append(Attention(self.input_size, self.embedding_size, prior_only=True))
            encdec_heads.append(Attention(self.input_size, self.embedding_size))
        # self attention
        self.attention_heads = nn.ModuleList(heads)

        self.multiheaded_selfattention = Parameter(torch.Tensor(self.embedding_size * attention_heads, self.input_size))

        self.attention_normalize = norm.LayerNorm(self.input_size)
            
        # encoder decoder

        self.encdec_heads = nn.ModuleList(encdec_heads)
        self.multiheaded_encdec = Parameter(torch.Tensor(self.embedding_size * attention_heads, self.input_size))
        self.encdec_normalize = norm.LayerNorm(self.input_size)
        # feedforward

        self.ffn = PointwiseFFN(self.input_size, self.hidden_size, self.input_size)

        self.ffn_normalize = norm.LayerNorm(self.input_size)

        self.init_weights()

    def init_weights(self):
    	for p in self.parameters():
        	if p.data.ndimension() >= 2:
        		nn.init.xavier_uniform_(p.data)
        	else:
        		nn.init.zeros_(p.data)

    def forward(self, x, encoded):
    	# the expected shape of x is (batch, sequence, feature)

        attention_z = []

        attention_scores = []
        attention_xqs = []
        attention_xvs = []
        attention_xks = []

        encdec_z = []

        encdec_scores = []
        encdec_xqs = []
        encdec_xvs = []
        encdec_xks = []


        # self attention
        for attention in self.attention_heads:
        	z, (scores, x_q, x_v, x_k) = attention(x)

        	attention_z.append(z)
        	attention_scores.append(scores)
        	attention_xqs.append(x_q)
        	attention_xvs.append(x_v)
        	attention_xks.append(x_k)
        
        concatenated_z = torch.cat(attention_z, 1)

        transformed_z = concatenated_z @ self.multiheaded_selfattention 

        normed_z = self.attention_normalize(transformed_z + x)

        # encoder-decoder attention

        for attention in self.encdec_heads:
            z, (scores, x_q, x_v, x_k) = attention(normed_z, encoded=encoded)

            encdec_z.append(z)
            encdec_scores.append(scores)
            encdec_xqs.append(x_q)
            encdec_xvs.append(x_v)
            encdec_xks.append(x_k)
        
        concatenated_encdec_z = torch.cat(encdec_z, 1)

        transformed_encdec_z = concatenated_encdec_z @ self.multiheaded_encdec 

        normed_encdec_z = self.encdec_normalize(transformed_encdec_z + normed_z)


        # ffn

        output = self.ffn(normed_encdec_z)

        normed_output = self.ffn_normalize(output + normed_encdec_z)
        
        return normed_output, (attention_z, attention_scores, attention_xqs, attention_xvs, attention_xks)

class LayeredDecoder(nn.Module):
    """Tokens must be passed into the network"""
    def __init__(self, input_size, embedding_sizes, hidden_sizes, attention_heads = None):
        super().__init__()
        self.input_size = input_size
        self.embedding_sizes = embedding_sizes
        self.hidden_sizes = hidden_sizes

        if attention_heads is None:
            self.attention_heads = [1]*len(embedding_sizes)
        elif type(attention_heads) is int:
            self.attention_heads = [attention_heads]*len(embedding_sizes)
        else:
            self.attention_heads = attention_heads

        self.layers = [Decoder(self.input_size, self.embedding_sizes[0], self.hidden_sizes[0], self.attention_heads[0])]

        for layer in range(1, len(self.embedding_sizes)):
            self.layers.append(Decoder(self.input_size, self.embedding_sizes[layer], self.hidden_sizes[layer], self.attention_heads[layer]))

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

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)
         
    def forward(self, x):
        return self.weight[:, :x.size(1), :] # (1, Seq, Feature)

class Transformer(nn.Module):
    """Tokens must be passed into the network"""
    def __init__(self, input_size, num_layers, encoder_hidden_sizes, decoder_hidden_sizes, encoder_attention_heads = None, decoder_attention_heads = None):
        super().__init__()
        self.input_size = input_size
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.decoder_hidden_sizes = decoder_hidden_sizes
        self.num_encoders = num_layers[0]
        self.num_decoders = num_layers[1]

        self.pos_embedding = PositionalEmbedding(self.input_size)

        self.encoders = LayeredEncoder(input_size, input_size, encoder_hidden_sizes, self.num_encoders, encoder_attention_heads)
        self.decoders = LayeredDecoder(input_size, input_size, decoder_hidden_sizes, self.num_decoders, decoder_attention_heads)


    def forward(self, x):
        # positional encoding
        pos_x = self.pos_embedding(x)

        # encoders
        # get last output - encoded
        # no input (zeros) and ask for first word
        # positional encoding from then on


if __name__ == '__main__':
    x = torch.rand(5,10)
    a = Attention(input_size=10, embedding_size=20, prior_only=True)
    e = Encoder(input_size = 10, embedding_size=20, hidden_size=40, attention_heads = 2)
    el = LayeredEncoder(input_size = 10, embedding_sizes=[10,10,10], hidden_sizes=[40,60,80])
    
    output, _ = el.forward(x)
    encoded = output[len(output)-1,:].reshape(1,-1)

    d = Decoder(input_size = 10, embedding_size=20,hidden_size=40, encoded=encoded, attention_heads=2)
    dl = LayeredDecoder(input_size = 10, embedding_sizes=[10,10,10], encoded=encoded, hidden_sizes=[40,60,80])
    output, _ = dl.forward(x)
    import pdb; pdb.set_trace()








#EOF
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

def prior_only_mask(dims):
    upper_tri_indices = np.triu_indices(dims, k=1)

    mask = torch.zeros(dims,dims)
    mask[upper_tri_indices] = 1
    return torch.Tensor(mask)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temp, attn_dropout=0.1):
        super().__init__()
        self.temp = temp
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # a = softmax(q . k^T / sqrt(dim)) * v
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temp

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, dropout=0.0):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_model/n_head
        self.d_v = d_model/n_head

        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_model/n_head)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_model/n_head)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_model/n_head)))

        self.attention = ScaledDotProductAttention(temp=np.power(d_model/n_head, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(d_model, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):
        # (batch size, sequence length, features)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        mask = mask.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

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

# class Attention(nn.Module):
#     """Tokens must be passed into the network"""
#     def __init__(self, input_size, embedding_size, prior_only=False):
#         super().__init__()

#         self.prior_only = prior_only

#         self.input_size = input_size
#         self.embedding_size = embedding_size

#         self.wq = Parameter(torch.Tensor(input_size, self.embedding_size))
#         self.wk = Parameter(torch.Tensor(input_size, self.embedding_size))
#        	self.wv = Parameter(torch.Tensor(input_size, self.embedding_size))

#         self.init_weights()

#     def init_weights(self):
#     	for p in self.parameters():
#         	if p.data.ndimension() >= 2:
#         		nn.init.xavier_uniform_(p.data)
#         	else:
#         		nn.init.zeros_(p.data)

#     def forward(self, x, encoded = None):
#     	# the expected shape of x is (batch, sequence, feature)

#         dk = math.sqrt(self.wk.shape[1])

#         if encoded is None:
#             Q = x @ self.wq
#             V = x @ self.wv
#             K = x @ self.wk
#         else:
#             Q = x @ self.wq
#             V = encoded @ self.wv
#             K = encoded @ self.wk


#         queried_keys = torch.bmm(Q, K.transpose(1,2))

#         if self.prior_only:
#         	upper_tri_indices = np.triu_indices(queried_keys.shape[0], k=1)

#         	inf_replace = torch.zeros(queried_keys.shape[0],queried_keys.shape[0])
#         	inf_replace[upper_tri_indices] = -np.inf

#         	queried_keys = queried_keys + inf_replace

#         scores = nn.functional.softmax(queried_keys/dk)
#         z = torch.bmm(scores, V)
        
#         return z, (scores, Q, V, K)

class Encoder(nn.Module):
    def __init__(self, d_model, hidden_size, n_heads = 1):
        super().__init__()

        if n_heads < 1:
        	print('attention_heads must be 1 or greater')
        	raise ValueError

        self.input_size = d_model
        self.hidden_size = hidden_size

        # self attention
        self.self_attention = MultiHeadAttention(n_heads, d_model)

        self.ffn = PointwiseFFN(self.input_size, self.hidden_size, self.input_size)

        self.ffn_normalize = norm.LayerNorm(self.input_size)

        self.init_weights()

    def init_weights(self):
    	for p in self.parameters():
        	if p.data.ndimension() >= 2:
        		nn.init.xavier_uniform_(p.data)
        	else:
        		nn.init.zeros_(p.data)

    def forward(self, input_seq):
        # batch, len, features
        output, attn = self.self_attention(input_seq, input_seq, input_seq)
    	
        forward_out = self.ffn(output)

        normed = self.ffn_normalize(output + forward_out)

        return normed

class LayeredEncoder(nn.Module):
    """Tokens must be passed into the network"""
    def __init__(self, input_size, hidden_sizes, attention_heads = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        if attention_heads is None:
        	self.attention_heads = [1]*len(embedding_sizes)
        elif type(attention_heads) is int:
            self.attention_heads = [attention_heads]*len(embedding_sizes)
        else:
            self.attention_heads = attention_heads

        self.layers = [Encoder(self.input_size, \
                                self.hidden_sizes[0], \
                                self.attention_heads[0])]

        for layer in range(1, len(self.attention_heads)):
        	self.layers.append(Encoder(self.input_size, self.hidden_sizes[layer], self.attention_heads[layer]))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, input_seq):
    	# the expected shape of x is (batch, sequence, feature)
    	output = input_seq
        # self attention
        for encoder in self.layers:
            output = encoder(output)
        # ffn
        return output
# Decoder
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_heads = 1):
        super().__init__()

        if attention_heads < 1:
        	print('attention_heads must be 1 or greater')
        	raise ValueError

        self.input_size = input_size
        self.hidden_size = hidden_size

        # self attention
        self.self_attention = MultiHeadAttention(n_heads, self.input_size)

        # encoder-decoder attention
        self.encdec_attention = MultiHeadAttention(n_heads, self.input_size)

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

    def forward(self, input_seq, encoded):
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

    def forward(self, encoded, target_seq):
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
            output, (z, scores, x_q, x_v, x_k) = self.layers[layer](target_seq, )

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


    def forward(self, x, out):
        # positional encoding
        pos_x = self.pos_embedding(x)

        # encoders
        enc_x = self.encoders(pos_x)
        
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
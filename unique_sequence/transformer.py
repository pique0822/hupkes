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
    # return torch.Tensor(mask).byte()

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
        self.d_k = int(d_model/n_head)
        self.d_v = int(d_model/n_head)

        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + int(d_model/n_head))))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + int(d_model/n_head))))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + int(d_model/n_head))))

        self.attention = ScaledDotProductAttention(temp=np.power(int(d_model/n_head), 0.5))
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

        if mask is not None:
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
        	self.attention_heads = [1]*len(hidden_sizes)
        elif type(attention_heads) is int:
            self.attention_heads = [attention_heads]*len(hidden_sizes)
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

        if n_heads < 1:
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
        prior_mask = prior_only_mask(input_seq.shape[1])

        output = input_seq
        output = self.self_attention(output, output, output, mask=prior_mask)
        encdec_output = self.encdec_attention(output, encoded, encoded)
        output = self.ffn(encdec_output)
        output = self.ffn_normalize(output + encdec_output)
        return output

class LayeredDecoder(nn.Module):
    """Tokens must be passed into the network"""
    def __init__(self, input_size, hidden_sizes, attention_heads = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        if attention_heads is None:
            self.attention_heads = [1]*len(hidden_sizes)
        elif type(attention_heads) is int:
            self.attention_heads = [attention_heads]*len(hidden_sizes)
        else:
            self.attention_heads = attention_heads

        self.layers = [Decoder(self.input_size, \
                                self.hidden_sizes[0], \
                                self.attention_heads[0])]

        for layer in range(1, len(self.hidden_sizes)):
            self.layers.append(Decoder(self.input_size, \
                                self.hidden_sizes[layer], \
                                self.attention_heads[layer]))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, input_seq, encoded):
        # the expected shape of x is (batch, sequence, feature)
        output = input_seq
        # self attention
        for decoder in self.layers:
            output = decoder(output, encoded)
        # ffn
        return output
        

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
        return self.weight[:, :x.size(1), :] # (batch, Seq, Feature)

class Transformer(nn.Module):
    """Tokens must be passed into the network"""
    def __init__(self, input_size, encoder_hidden_sizes, decoder_hidden_sizes, encoder_attention_heads = None, decoder_attention_heads = None):
        super().__init__()
        self.input_size = input_size
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.decoder_hidden_sizes = decoder_hidden_sizes

        self.pos_embedding = PositionalEmbedding(self.input_size)

        self.encoders = LayeredEncoder(input_size, encoder_hidden_sizes, encoder_attention_heads)
        self.decoders = LayeredDecoder(input_size, decoder_hidden_sizes, decoder_attention_heads)


    def forward(self, input_seq, target_seq):
        # positional encoding
        pos_input = self.pos_embedding(input_seq)
        # import pdb; pdb.set_trace()

        # encoders
        enc_input = self.encoders(pos_input)
        # import pdb; pdb.set_trace()

        pos_target = self.pos_embedding(target_seq)
        import pdb; pdb.set_trace()

        dec_output = self.decoders(pos_target, enc_input)
        # import pdb; pdb.set_trace()

        return dec_output


if __name__ == '__main__':
    x = torch.rand(1,5,2)
    y = torch.rand(1,2,2)
    tr = Transformer(2, [40], [40])

    tr(x,y)








#EOF
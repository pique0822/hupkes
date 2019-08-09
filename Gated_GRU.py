import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from enum import IntEnum

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class GatedGRU(nn.Module):
    """Tokens must be passed into the network"""
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_size)

        # update gate
        self.W_iu = Parameter(torch.Tensor(embedding_size, hidden_size))
        self.W_hu = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_u = Parameter(torch.Tensor(hidden_size))

        # reset gate
        self.W_ir = Parameter(torch.Tensor(embedding_size, hidden_size))
        self.W_hr = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_r = Parameter(torch.Tensor(hidden_size))

        # hidden state
        self.W_ih = Parameter(torch.Tensor(embedding_size, hidden_size))
        self.W_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = Parameter(torch.Tensor(hidden_size))

        self.decoder = nn.Linear(hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def init_hidden(self):
        h_t = torch.zeros(self.hidden_size)
        return h_t

    def forward(self, x, init_state):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()

        update_gates = []
        reset_gates = []
        hidden_states = []

        h_t = init_state

        for t in range(seq_sz): # iterate over the time steps
            x_t = x[:, t, :]
            x_t = self.embedding(x_t)

            z_t = torch.sigmoid(x_t @ self.W_iu + h_t @ self.W_hu + self.b_u)

            r_t = torch.sigmoid(x_t @ self.W_ir + h_t @ self.W_hr + self.b_r)

            h_t = (1 - z_t) * h_t + z_t * torch.tanh(x_t @ self.W_ih + (r_t * h_t) @ self.W_hh + self.b_h)


            update_gates.append(z_t.unsqueeze(Dim.batch))
            reset_gates.append(r_t.unsqueeze(Dim.batch))
            hidden_states.append(h_t.unsqueeze(Dim.batch))


        # hidden_seq = torch.cat(hidden_states, dim=Dim.batch)
        # # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        # hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()

        output = self.decoder(h_t.reshape(-1,self.hidden_size))

        return output, h_t, (update_gates, reset_gates, hidden_states)

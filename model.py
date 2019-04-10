import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from layers import GraphConvolution


class Classify(torch.nn.Module):
    def __init__(self, params, vocab_size, ntags, pte=None):
        super(Classify, self).__init__()
        self.params = params
        self.word_embeddings = nn.Embedding(vocab_size, params.emb_dim)
        if pte is None:
            nn.init.xavier_uniform_(self.word_embeddings.weight)
        else:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pte))
        self.text_encoder = CnnEncoder(
            params.filters, params.emb_dim, params.kernel_size) if params.encoder == 1 else LstmEncoder(
            params.hidden_dim, params.emb_dim)
        self.dropout = nn.Dropout(params.dropout)
        if params.encoder == 2:
            self.gcn1 = GraphConvolution(params.hidden_dim, params.node_emb_dim, params.dropout, act=F.relu)
            self.linear_transform = nn.Linear(in_features=params.node_emb_dim,
                                              out_features=ntags)
        else:
            self.linear_transform = nn.Linear(in_features=params.hidden_dim,
                                              out_features=ntags)

    def forward(self, input_sents, input_lens):
        embeds = self.word_embeddings(input_sents)  # bs * max_seq_len * emb
        h = self.text_encoder(embeds, input_lens)  # bs * 100 * hidden
        h = self.dropout(F.relu(h))  # Relu activation and dropout
        if self.params.encoder == 2:
            # Currently it's a dummy matrix with all edge weights one
            adj_matrix = torch.ones((h.size(0), h.size(0)))
            h = self.gcn1(h, adj_matrix)
            # Simple max pool on all node representations
            h, _ = h.max(dim=0)
        h = self.linear_transform(h)  # bs * ntags
        return h


class LstmEncoder(torch.nn.Module):
    def __init__(self, hidden_dimension, embedding_dimension):
        super(LstmEncoder, self).__init__()
        self.hidden_dim = hidden_dimension
        self.lstm = nn.LSTM(embedding_dimension, hidden_dimension)

    def forward(self, embeds, seq_lens):
        # By default a LSTM requires the batch_size as the second dimension
        # You could also use batch_first=True while declaring the LSTM module, then this permute won't be required
        embeds = embeds.permute(1, 0, 2)  # seq_len * batch_size * embedding_dim

        packed_input = pack_padded_sequence(embeds, seq_lens)
        _, (hn, cn) = self.lstm(packed_input)
        # two outputs are returned. _ stores all the hidden representation at each time_step
        # (hn, cn) is just for convenience, and is hidden representation and context after the last time_step
        # _ : will be of PackedSequence type, once unpacked, you will get a tensor of size: seq_len x bs x hidden_dim
        # hn : 1 x bs x hidden_dim

        return hn[-1]  # bs * hidden_dim


class CnnEncoder(torch.nn.Module):
    def __init__(self, filters, emb_dim, kernel_size):
        super(CnnEncoder, self).__init__()
        self.conv_tri = nn.Conv1d(in_channels=emb_dim, out_channels=filters, kernel_size=kernel_size, padding=1)

    def forward(self, embeds, seq_lens):
        embeds = embeds.permute(0, 2, 1)  # bs * ed * seq
        h = self.conv_tri(embeds)  # bs * hd * seq
        # Max pooling
        h = h.max(dim=2)[0]  # bs * hd
        h = F.relu(h)
        return h  # bs * hidden_dim

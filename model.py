import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from layers import GraphConvolution, GraphAttentionLayer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm


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
        elif params.encoder == 3:
            self.gcn1 = GraphConvolution(params.hidden_dim, params.node_emb_dim, params.dropout, act=F.relu)
            # Add the attention thingy
            self.linear_transform = nn.Linear(in_features=params.node_emb_dim,
                                              out_features=ntags)
        elif params.encoder == 4:
            self.gcn1 = GraphAttentionLayer(params.hidden_dim, params.node_emb_dim, params.dropout, 0.2)
            self.attentions = [GraphAttentionLayer(params.hidden_dim, params.node_emb_dim, dropout=params.dropout,
                                                   alpha=0.2, concat=True) for _ in range(0)]
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_{}'.format(i), attention)

            self.out_att = GraphAttentionLayer(params.hidden_dim, params.node_emb_dim, dropout=params.dropout,
                                               alpha=0.2, concat=False)
            # Add the attention thingy
            self.linear_transform = nn.Linear(in_features=params.node_emb_dim,
                                              out_features=ntags)
        elif params.encoder == 5:
            self.gcn1 = GraphAttentionLayer(params.hidden_dim, params.node_emb_dim, params.dropout, 0.2)
            self.attentions = [GraphAttentionLayer(params.hidden_dim, params.node_emb_dim, dropout=params.dropout,
                                                   alpha=0.2, concat=True) for _ in range(2)]
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_{}'.format(i), attention)

            self.out_att = GraphAttentionLayer(params.node_emb_dim * 2, params.node_emb_dim, dropout=params.dropout,
                                               alpha=0.2, concat=False)
            # Add the attention thingy
            self.linear_transform = nn.Linear(in_features=params.node_emb_dim,
                                              out_features=ntags)
        else:
            self.linear_transform = nn.Linear(in_features=params.hidden_dim,
                                              out_features=ntags)

    def forward(self, input_sents, input_lens, adj=None, actual_sentence=None):
        embeds = self.word_embeddings(input_sents)  # bs * max_seq_len * emb
        h = self.text_encoder(embeds, input_lens)  # bs * 100 * hidden
        h = self.dropout(F.relu(h))  # Relu activation and dropout
        if self.params.encoder == 2:
            # Currently it's a dummy matrix with all edge weights one
            adj_matrix = np.ones((h.size(0), h.size(0))) if adj is None else adj
            np.fill_diagonal(adj_matrix, 0)
            adj_matrix = self.to_tensor(adj_matrix)
            h = self.gcn1(h, adj_matrix)
            # Simple max pool on all node representations
            h, _ = h.max(dim=0)
        elif self.params.encoder == 3:
            # Currently it's a dummy matrix with all edge weights one
            adj_matrix = np.ones((h.size(0), h.size(0))) if adj is None else adj
            # Setting link between same sentences to 0
            np.fill_diagonal(adj_matrix, 0)
            adj_matrix = self.to_tensor(adj_matrix)
            h = self.gcn1(h, adj_matrix)                    # num_sentences * node_emb_dim
            # Adding self attention layer on the representations
            att = F.softmax(torch.mm(h, h.transpose(0, 1)) / np.sqrt(self.params.node_emb_dim), dim=1)

            if self.params.plot == 1:
                mat = np.matrix(att.data.numpy())
                fig = plt.figure()
                im = plt.imshow(mat, interpolation='nearest', cmap=cm.hot, origin='lower')
                plt.xlabel('Sentence Number')
                plt.ylabel('Sentence Number')
                fig.colorbar(im)
                if mat.shape[0] < 10:
                    plt.xticks(range(0, mat.shape[0], 1))
                    plt.yticks(range(0, mat.shape[0], 1))
                fig.savefig('plots/sample_attn_{}.png'.format(mat.shape[0]))
            # Simple max pool on all node representations
            h, _ = h.max(dim=0)
        elif self.params.encoder == 4:
            # Currently it's a dummy matrix with all edge weights one
            adj_matrix = np.ones((h.size(0), h.size(0))) if adj is None else adj
            # Setting link between same sentences to 0
            np.fill_diagonal(adj_matrix, 0)
            adj_matrix = self.to_tensor(adj_matrix)

            h = F.dropout(h, self.params.dropout, training=self.training)
            h, attn = self.out_att(h, adj_matrix)
            h = F.elu(h)
            if self.params.plot == 1:

                mat = np.matrix(adj_matrix.cpu().data.numpy())
                fig = plt.figure()
                im = plt.imshow(mat, interpolation='nearest', cmap=cm.hot, origin='lower')
                plt.xlabel('Sentence Number')
                # for j, actual_sent in enumerate(actual_sentence):
                #    plt.text(10, 2 + j, actual_sent, ha='right', wrap=True, size=2)
                plt.ylabel('Sentence Number')
                if mat.shape[0] < 10:
                    plt.xticks(range(0, mat.shape[0], 1))
                    plt.yticks(range(0, mat.shape[0], 1))
                fig.colorbar(im)
                fig.savefig('plots/adj/sample_adj_matrix_{}.png'.format(mat.shape[0]))

                mat = np.matrix(attn.cpu().data.numpy())
                fig = plt.figure()
                im = plt.imshow(mat, interpolation='nearest', cmap=cm.hot, origin='lower')
                plt.xlabel('Sentence Number')
                #for j, actual_sent in enumerate(actual_sentence):
                #    plt.text(10, 2 + j, actual_sent, ha='right', wrap=True, size=2)
                plt.ylabel('Sentence Number')
                if mat.shape[0] < 10:
                    plt.xticks(range(0, mat.shape[0], 1))
                    plt.yticks(range(0, mat.shape[0], 1))
                fig.colorbar(im)
                fig.savefig('plots/adj/sample_attn_gat_{}.png'.format(mat.shape[0]))
                if actual_sentence is not None:
                    file = open('plots/adj/{}.txt'.format(mat.shape[0]), 'w')
                    for actual_sent in actual_sentence:
                        file.write(actual_sent + "\n")
                    file.close()

            # Simple max pool on all node representations
            h, _ = h.max(dim=0)
        elif self.params.encoder == 5:
            # Currently it's a dummy matrix with all edge weights one
            adj_matrix = np.ones((h.size(0), h.size(0))) if adj is None else adj
            # Setting link between same sentences to 0
            np.fill_diagonal(adj_matrix, 0)
            adj_matrix = self.to_tensor(adj_matrix)

            hs = []
            for i, att in enumerate(self.attentions):
                h_i, att_i = att(h, adj_matrix)
                if self.params.plot == 1:
                    mat = np.matrix(att_i.cpu().data.numpy())
                    fig = plt.figure()
                    im = plt.imshow(mat, interpolation='nearest', cmap=cm.hot, origin='lower')
                    plt.xlabel('Sentence Number')
                    if mat.shape[0] < 10:
                        plt.xticks(range(0, mat.shape[0], 1))
                        plt.yticks(range(0, mat.shape[0], 1))
                    #for j, actual_sent in enumerate(actual_sentence):
                    #    plt.text(10, 2 + j, actual_sent, ha='right', wrap=True, size=2)
                    plt.ylabel('Sentence Number')
                    fig.colorbar(im)
                    fig.savefig('plots/sample_attn_gat_{}_{}.png'.format(i, mat.shape[0]))
                hs.append(h_i)
            h = torch.cat(hs, dim=1)
            h = F.dropout(h, self.params.dropout, training=self.training)
            h, attn = self.out_att(h, adj_matrix)
            h = F.elu(h)
            if self.params.plot == 1:

                mat = np.matrix(adj_matrix.cpu().data.numpy())
                fig = plt.figure()
                im = plt.imshow(mat, interpolation='nearest', cmap=cm.hot, origin='lower')
                plt.xlabel('Sentence Number')
                # for j, actual_sent in enumerate(actual_sentence):
                #    plt.text(10, 2 + j, actual_sent, ha='right', wrap=True, size=2)
                plt.ylabel('Sentence Number')
                if mat.shape[0] < 10:
                    plt.xticks(range(0, mat.shape[0], 1))
                    plt.yticks(range(0, mat.shape[0], 1))
                fig.colorbar(im)
                fig.savefig('plots/adj/sample_adj_matrix_{}.png'.format(mat.shape[0]))

                mat = np.matrix(attn.cpu().data.numpy())
                fig = plt.figure()
                im = plt.imshow(mat, interpolation='nearest', cmap=cm.hot, origin='lower')
                plt.xlabel('Sentence Number')
                #for j, actual_sent in enumerate(actual_sentence):
                #    plt.text(10, 2 + j, actual_sent, ha='right', wrap=True, size=2)
                plt.ylabel('Sentence Number')
                if mat.shape[0] < 10:
                    plt.xticks(range(0, mat.shape[0], 1))
                    plt.yticks(range(0, mat.shape[0], 1))
                fig.colorbar(im)
                fig.savefig('plots/adj/sample_attn_gat_{}.png'.format(mat.shape[0]))
                if actual_sentence is not None:
                    file = open('plots/adj/{}.txt'.format(mat.shape[0]), 'w')
                    for actual_sent in actual_sentence:
                        file.write(actual_sent + "\n")
                    file.close()

            # Simple max pool on all node representations
            h, _ = h.max(dim=0)
        h = self.linear_transform(h)  # bs * ntags
        return h
    
    @staticmethod
    def to_tensor(arr):
        # list -> Tensor (on GPU if possible)
        if torch.cuda.is_available():
            tensor = torch.tensor(arr).type(torch.cuda.FloatTensor)
        else:
            tensor = torch.tensor(arr).type(torch.FloatTensor)
        return tensor


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

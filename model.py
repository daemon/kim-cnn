import random

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class KimCNN(nn.Module):
    def __init__(self, word_model, **config):
        super().__init__()
        n_fmaps = config.get("n_feature_maps", 100)
        weight_lengths = config.get("weight_lengths", [3, 4, 5])
        embedding_dim = word_model.dim

        self.word_model = word_model
        n_c = word_model.n_channels
        self.conv_layers = [nn.Conv2d(n_c, n_fmaps, (w, embedding_dim), padding=(w - 1, 0)) for w in weight_lengths]
        for i, conv in enumerate(self.conv_layers):
            self.add_module("conv{}".format(i), conv)
        self.dropout = nn.Dropout(config.get("dropout", 0.5))
        self.fc = nn.Linear(len(self.conv_layers) * n_fmaps, config.get("n_labels", 5))

    def preprocess(self, sentences):
        return torch.from_numpy(np.array(self.word_model.lookup(sentences)))

    def compute_corr_matrix(self, output):
        grads = autograd.grad(output, self.sent_weights)
        grads = grads[0].squeeze(0).squeeze(0)
        # return np.cov(grads.cpu().data.numpy())
        sz = self.sent_weights.size(2)
        corr_matrix = np.empty((sz, sz))
        for i, g1 in enumerate(grads):
            for j, g2 in enumerate(grads):
                corr_matrix[i, j] = torch.dot(g1, g2).cpu().data[0]
        return corr_matrix

    def compute_grad_norm(self, output):
        grad_norms = autograd.grad(output, self.sent_weights, create_graph=True)
        grad_norms = grad_norms[0].squeeze(0).squeeze(0)
        grad_norms = [torch.sqrt(torch.sum(g**2)) for g in grad_norms]
        return torch.cat(grad_norms).cpu().data.numpy()

    def rank(self, sentence):
        m_in = autograd.Variable(torch.Tensor(self.word_model.lookup([sentence])).long().cuda())
        m_out = self.forward(m_in)
        return torch.max(m_out, 1)[1], m_out

    def loss(self):
        return 0
        def conv_sim_loss(weights, n=5):
            weights = random.sample(weights, 2 * n)
            weights1, weights2 = weights[:n], weights[n:]
            loss = 0
            tot = 0
            for i, w1 in enumerate(weights1):
                for j in range(i + 1, n):
                    w2 = weights2[j]
                    loss += compute_sim_loss(w1, w2, 5E-3)
                    tot += 1
            return loss / tot
        conv_layers = [c.weight for c in self.conv_layers]
        loss = conv_sim_loss(conv_layers[0].split(1, 0)) + conv_sim_loss(conv_layers[1].split(1, 0)) + \
            conv_sim_loss(conv_layers[2].split(1, 0))
        return loss

    def forward(self, x):
        self.sent_weights = x = self.word_model(x) # shape: (batch, channel, sent length, embed dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv_layers]
        x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)

class MultiChannelWordModel(nn.Module):
    def __init__(self, id_dict, weights, unknown_vocab=[]):
        super().__init__()
        self.n_channels = 2
        self.non_static_model = SingleChannelWordModel(id_dict, weights, unknown_vocab, static=False)
        self.static_model = SingleChannelWordModel(id_dict, self.non_static_model.weights)
        self.dim = self.static_model.dim

    def forward(self, x):
        batch1 = self.static_model(x)
        batch2 = self.non_static_model(x)
        return torch.cat((batch1, batch2), dim=1)

    def lookup(self, sentences):
        return self.static_model.lookup(sentences)

class SingleChannelWordModel(nn.Module):
    def __init__(self, id_dict, weights, unknown_vocab=[], static=True):
        super().__init__()
        vocab_size = len(id_dict) + len(unknown_vocab)
        self.n_channels = 1
        self.lookup_table = id_dict
        last_id = max(id_dict.values())
        for word in unknown_vocab:
            last_id += 1
            self.lookup_table[word] = last_id
        self.weights = np.concatenate((weights, np.random.rand(len(unknown_vocab), 300) - 0.5))
        self.dim = self.weights.shape[1]
        self.embedding = nn.Embedding(vocab_size, self.dim, padding_idx=2)
        self.embedding.weight.data.copy_(torch.from_numpy(self.weights))
        if static:
            self.embedding.weight.requires_grad = False

    @classmethod
    def make_random_model(cls, id_dict, unknown_vocab=[], dim=300):
        weights = np.random.rand(len(id_dict), dim) - 0.5
        return cls(id_dict, weights, unknown_vocab, static=False)

    def forward(self, x):
        batch = self.embedding(x)
        return batch.unsqueeze(1)

    def lookup(self, sentences):
        indices_list = []
        max_len = 0
        for sentence in sentences:
            indices = []
            for word in str(sentence).split():
                try:
                    index = self.lookup_table[word]
                    indices.append(index)
                except KeyError:
                    continue
            indices_list.append(indices)
            if len(indices) > max_len:
                max_len = len(indices)
        for indices in indices_list:
            indices.extend([2] * (max_len - len(indices)))
        return indices_list


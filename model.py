import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_func

class KimCNN(nn.Module):
    def __init__(self, word_model, **config):
        super().__init__()
        n_feat_maps = config.get("n_feature_maps", 100)
        weight_lengths = config.get("weight_lengths", [3, 4, 5])
        embedding_dim = word_model.dim

        self.word_model = word_model
        self.conv_layers = [nn.Conv2d(1, n_feat_maps, (w, embedding_dim), padding=(w - 1, 0)) for w in weight_lengths]
        for i, conv in enumerate(self.conv_layers):
            self.add_module("conv{}".format(i), conv)
        self.dropout = nn.Dropout(config.get("dropout", 0.3))
        self.fc = nn.Linear(len(self.conv_layers) * n_feat_maps, config.get("n_labels", 5))

    def preprocess(self, sentences):
        return torch.from_numpy(np.array(self.word_model.lookup(sentences)))

    def forward(self, x):
        x = self.word_model(x) # shape: (batch, channel, sent length, embed dim)
        x = [nn_func.relu(conv(x)).squeeze(3) for conv in self.conv_layers]
        x = [nn_func.max_pool1d(c, c.size(2)).squeeze(2) for c in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)

class SingleChannelWordModel(nn.Module):
    def __init__(self, id_dict, weights, static=True):
        super().__init__()
        vocab_size = len(id_dict)
        self.lookup_table = id_dict
        self.dim = weights.shape[1]
        self.embedding = nn.Embedding(vocab_size, self.dim, padding_idx=2) # pytorch 0.1x bug
        self.embedding.weight.data.copy_(torch.from_numpy(weights))
        if static:
            self.embedding.weight.requires_grad = False

    @classmethod
    def make_random_model(cls, id_dict, dim=300):
        weights = np.random.rand(len(id_dict), dim) * 2 - 1
        return cls(id_dict, weights, static=False)

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


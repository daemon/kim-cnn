import torch
import torch.nn as nn
import model
import numpy as np
import os
import pickle

def load_sst_sets(dirname="data", fmt="stsa.fine.{}.tsv"):
    set_names = ["phrases.train", "dev", "test"]
    def read_set(name):
        data_set = []
        with open(os.path.join(dirname, fmt.format(name))) as f:
            for line in f.readlines():
                sentiment, sentence = line.replace("\n", "").split("\t")
                data_set.append((sentiment, sentence))
        return np.array(data_set)
    return [read_set(name) for name in set_names]

def load_embed_data(dirname="data", weights_file="embed_weights.npy", id_file="word_id.dat"):
    id_file = os.path.join(dirname, id_file)
    weights_file = os.path.join(dirname, weights_file)
    with open(id_file, "rb") as f:
        id_dict = pickle.load(f)
    with open(weights_file, "rb") as f:
        weights = np.load(f)
    return (id_dict, weights)

def clip_weights(parameter, s=3):
    norm = abs(parameter.weight.data.norm())
    if norm < s:
        return
    parameter.weight.data.mul_(s / norm)

def main():
    torch.cuda.set_device(1)
    word_model = model.MultiChannelWordModel(*load_embed_data())
    word_model.cuda()
    kcnn = model.KimCNN(word_model)
    kcnn.cuda()
    criterion = nn.CrossEntropyLoss()
    parameters = filter(lambda p: p.requires_grad, kcnn.parameters())
    optimizer = torch.optim.Adadelta(parameters, lr=0.001, weight_decay=0.)

    train_set, dev_set, test_set = load_sst_sets()
    for epoch in range(30):
        kcnn.train()
        optimizer.zero_grad()
        np.random.shuffle(train_set)
        mbatch_size = 50
        i = 0
        while i + mbatch_size < len(train_set):
            mbatch = train_set[i:i + mbatch_size]
            train_in = mbatch[:, 1].reshape(-1)
            train_out = mbatch[:, 0].flatten().astype(np.int)
            train_out = torch.autograd.Variable(torch.from_numpy(train_out))
            train_in = kcnn.preprocess(train_in)

            train_in = torch.autograd.Variable(train_in.cuda())
            train_out = train_out.cuda()

            scores = kcnn(train_in)
            loss = criterion(scores, train_out)
            loss.backward()
            #nn.utils.clip_grad_norm(parameters, 0.1)
            optimizer.step()
            for conv_layer in kcnn.conv_layers:
                clip_weights(conv_layer)
            i += mbatch_size

            if i % 10000 == 0:
                kcnn.eval()
                dev_in = test_set[:, 1].reshape(-1)
                dev_out = test_set[:, 0].flatten().astype(np.int)
                dev_out = torch.autograd.Variable(torch.from_numpy(dev_out))
                dev_in = kcnn.preprocess(dev_in)
                dev_in = torch.autograd.Variable(dev_in.cuda())
                dev_out = dev_out.cuda()
                scores = kcnn(dev_in)
                n_correct = (torch.max(scores, 1)[1].view(len(test_set)).data == dev_out.data).sum()
                print("Accuracy: {}".format(n_correct / len(test_set)))
                kcnn.train()

if __name__ == "__main__":
    main()

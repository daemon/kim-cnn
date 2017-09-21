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

def convert_dataset(model, dataset):
    model_in = dataset[:, 1].reshape(-1)
    model_out = dataset[:, 0].flatten().astype(np.int)
    model_out = torch.autograd.Variable(torch.from_numpy(model_out)).cuda()
    model_in = model.preprocess(model_in)
    model_in = torch.autograd.Variable(model_in.cuda())
    return (model_in, model_out)

def main():
    torch.cuda.set_device(1)
    word_model = model.SingleChannelWordModel.make_random_model(load_embed_data()[0])
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
            train_in, train_out = convert_dataset(kcnn, mbatch)

            scores = kcnn(train_in)
            loss = criterion(scores, train_out)
            loss.backward()
            optimizer.step()
            for conv_layer in kcnn.conv_layers:
                clip_weights(conv_layer)
            i += mbatch_size

            if i % 10000 == 0:
                kcnn.eval()
                dev_in, dev_out = convert_dataset(kcnn, dev_set)
                scores = kcnn(dev_in)
                n_correct = (torch.max(scores, 1)[1].view(len(dev_set)).data == dev_out.data).sum()
                accuracy = n_correct / len(dev_set)
                print("Dev set accuracy: {}".format(accuracy))
                if accuracy > 0.45:
                    test_in, test_out = convert_dataset(kcnn, test_set)
                    scores = kcnn(test_in)
                    n_correct = (torch.max(scores, 1)[1].view(len(test_set)).data == test_out.data).sum()
                    accuracy = n_correct / len(test_set)
                    print("Test set accuracy: {}".format(accuracy))
                    return
                kcnn.train()

if __name__ == "__main__":
    main()

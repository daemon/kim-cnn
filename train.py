import torch
import model
import numpy as np
import os
import spacy

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

def one_hot_vec_factory(size=5):
    eye = np.eye(size)
    def make_one_hot_vec(index):
        return eye[int(index)]
    return make_one_hot_vec

def main():
    torch.cuda.set_device(1)
    word_model = model.StaticWordModel(spacy.load("en_core_web_md"))
    word_model.cuda()
    kcnn = model.KimCNN(word_model)
    kcnn.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    parameters = filter(lambda p: p.requires_grad, kcnn.parameters())
    optimizer = torch.optim.Adadelta(parameters, lr=0.05, weight_decay=0.)

    train_set, dev_set, test_set = load_sst_sets()
    for epoch in range(30):
        kcnn.train()
        optimizer.zero_grad()
        np.random.shuffle(train_set)
        mbatch_size = 500
        i = 0
        while i + mbatch_size < len(train_set):
            mbatch = train_set[i:i + mbatch_size]
            train_in = mbatch[:, 1].reshape(-1)
            train_out = mbatch[:, 0].flatten().astype(np.int)
            train_out = torch.autograd.Variable(torch.from_numpy(train_out))
            train_in = kcnn.preprocess(train_in)

            train_in = train_in.cuda()
            train_out = train_out.cuda()

            scores = kcnn(train_in)
            loss = criterion(scores, train_out)
            loss.backward()
            optimizer.step()
            i += mbatch_size

            if i % 1000 == 0:
                kcnn.eval()
                dev_in = dev_set[:, 1].reshape(-1)
                dev_out = dev_set[:, 0].flatten().astype(np.int)
                dev_out = torch.autograd.Variable(torch.from_numpy(dev_out))
                dev_in = kcnn.preprocess(dev_in)
                dev_in = dev_in.cuda()
                dev_out = dev_out.cuda()
                scores = kcnn(dev_in)
                n_correct = (torch.max(scores, 1)[1].view(len(dev_set)).data == dev_out.data).sum()
                print("Accuracy: {}".format(n_correct / len(dev_set)))
                kcnn.train()

if __name__ == "__main__":
    main()

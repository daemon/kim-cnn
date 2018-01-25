import argparse

import numpy as np
import torch

import model

def run_grad_weight(kcnn, sentence):
    out = kcnn.rank(sentence)
    h = kcnn.compute_grad_norm(torch.sum(out[1]))
    for w, score in zip(sentence.split(), h):
        print(w, score)

def run_grad_pca(kcnn, sentence):
    out = kcnn.rank(sentence)
    print("Prediction: {}".format(out[0].cpu().data[0]))
    toks = sentence.split()
    mat = kcnn.compute_corr_matrix(torch.sum(out[1]))
    max_len = max([len(tok) for tok in toks])
    fmt_str = " " * (max_len + 1) + " ".join(["{:>%s}" % max(len(word), 6) for word in toks])
    print(fmt_str.format(*toks))
    for i, (word, row) in enumerate(zip(toks, mat)):
        print(("{:>%s}" % max_len).format(word), end=" ")
        for j, (w2, val) in enumerate(zip(toks, row)):
            if i == j and abs(val) > 0.1:
                print("\x1b[1;33m", end="")
            print(("{:>%s}" % max(len(w2), 6)).format(round(val, 3)), end=" ")
            print("\x1b[1;0m", end="")
        print()

    s, v = np.linalg.eig(mat)
    fmt_str = " ".join(["{:>%s}" % max(len(word), 6) for word in toks])
    v = v.transpose()
    print(fmt_str.format(*toks) + " [lambda]")
    for row, s_val in zip(v, s):
        for word, val in zip(toks, row):
            if abs(val) > 0.25:
                print("\x1b[1;33m", end="")
            print(("{:>%s}" % max(len(word), 6)).format(round(val, 3)), end=" ")
            print("\x1b[1;0m", end="")
        print(round(s_val, 3))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", type=str)
    parser.add_argument("--input_file", type=str, default="model.pt")
    parser.add_argument("--mode", type=str, choices=["grad_pca", "grad_weight"], default="grad_pca")
    args = parser.parse_args()
    kcnn = torch.load(args.input_file)
    kcnn.eval()
    if args.mode == "grad_pca":
        run_grad_pca(kcnn, args.sentence)
    else:
        run_grad_weight(kcnn, args.sentence)

main()
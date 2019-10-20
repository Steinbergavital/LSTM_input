import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import sys

"""
Blog post:
Taming LSTMs: Variable-sized mini-batches and why PyTorch is good for your health:
https://medium.com/@_willfalcon/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
"""

sent_1_x = ['is', 'it', 'too', 'late', 'now', 'say', 'sorry']
sent_1_y = ['VB', 'PRP', 'RB', 'RB', 'RB', 'VB', 'JJ']

sent_2_x = ['ooh', 'ooh']
sent_2_y = ['NNP', 'NNP']

sent_3_x = ['sorry', 'yeah']
sent_3_y = ['JJ', 'NNP']

X = [sent_1_x, sent_2_x, sent_3_x]
Y = [sent_1_y, sent_2_y, sent_3_y]

print('X: ', X)
print('Y: ', Y)

# map sentences to vocab dictionary:
vocab = {'<PAD>': 0, 'is': 1, 'it': 2, 'too': 3, 'late': 4, 'now': 5, 'say': 6, 'sorry': 7, 'ooh': 8, 'yeah': 9}

# fancy nested list comprehension
X = [[vocab[word] for word in sentence] for sentence in X]

# X now looks like:
# [[1, 2, 3, 4, 5, 6, 7], [8, 8], [7, 9]]

tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}

# fancy nested list comprehension
Y = [[tags[tag] for tag in sentence] for sentence in Y]
print('X: ', X)
print('Y: ', Y)

# get the length of each sentence
X_lengths = [len(sentence) for sentence in X]
print("X_lengths: ", X_lengths)

# create an empty matrix with padding tokens
pad_token = vocab['<PAD>']
longest_sent = max(X_lengths)
batch_size = len(X)
padded_X = np.ones((batch_size, longest_sent)) * pad_token
print("padded x: ", padded_X)
print("Batch size: ", batch_size)

mask = np.ones((batch_size, longest_sent)) * pad_token

# copy over the actual sequences
for i, x_len in enumerate(X_lengths):
  sequence = X[i]
  padded_X[i, 0:x_len] = sequence[:x_len]
  mask[i, 0:x_len] = 1

Y = [[1, 2, 3, 3, 3, 1, 4],
    [5, 5],
    [4, 5]]

# get the length of each sentence
Y_lengths = [len(sentence) for sentence in Y]

# create an empty matrix with padding tokens
pad_token = tags['<PAD>']
longest_sent = max(Y_lengths)
batch_size = len(Y)
padded_Y = np.ones((batch_size, longest_sent)) * pad_token
print("pad_token: ", pad_token)
print("longest_sent: ", longest_sent)
print("padded x: ", padded_X)
print("Batch size: ", batch_size)

# copy over the actual sequences
for i, y_len in enumerate(Y_lengths):
  sequence = Y[i]
  padded_Y[i, 0:y_len] = sequence[:y_len]

print("padded y: ", padded_Y)
print("mask: ", mask)
print("vocab size: ", len(vocab))
embeds = nn.Embedding(len(vocab), 3,padding_idx=0)
print("embeds: ", embeds)
xemb = embeds(torch.tensor(padded_X, dtype=torch.long))
print("embeddings: ", xemb)
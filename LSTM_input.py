import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import sys
import torch.optim as optim
import torch.autograd as autograd

"""
Based on the following blog post:
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
batch_size = 3
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
vocab_size = len(vocab)
embeds = nn.Embedding(len(vocab), 5,padding_idx=0)
print("embeds: ", embeds)
xemb = embeds(torch.tensor(padded_X, dtype=torch.long))
print("embeddings: ", xemb.shape)
#xemb = xemb.permute(1,2,0)
#print("embeddings: ", xemb.shape)

class BieberLSTM(nn.Module):
    def __init__(self, xlens, nb_layers, tgs, nb_lstm_units=100, embedding_dim=5, batch_size=3):
        super(BieberLSTM, self).__init__()
        self.hidden_dim = nb_lstm_units
        self.batch_size = batch_size
        self.nb_layers = nb_layers
        self.tags = tgs
        self.embedding_dim = embedding_dim
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size= self.hidden_dim,num_layers=self.nb_layers,batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.nb_tags = len(self.tags) - 1
        #self.linear = nn.Linear(self.hidden_dim, self.nb_tags)
        self.linear = nn.Linear(self.hidden_dim, 1)
        self.xlengths = xlens


    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        print("hidden dim: ", self.hidden_dim)
        hidden_a = torch.randn(self.nb_layers, self.batch_size, self.hidden_dim)
        hidden_b = torch.randn(self.nb_layers, self.batch_size, self.hidden_dim)
        return (hidden_a,hidden_b)


    def forward(self, paddedX):
        output = torch.nn.utils.rnn.pack_padded_sequence(paddedX, self.xlengths, batch_first=True)
        output, hidden_out = self.lstm(output,self.init_hidden())
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        tag_space = self.linear(output)
        #tag_scores = F.log_softmax(tag_space, dim=1)
        tag_scores = torch.sigmoid(tag_space)
        return tag_scores

HIDDEN_DIM = 3
EMBEDDING_DIM = 5
NB_LSTM_UNITS = 100
NUM_LAYERS = 2
model = BieberLSTM(X_lengths, NUM_LAYERS, tags, NB_LSTM_UNITS, EMBEDDING_DIM, batch_size)
h0 = torch.randn(EMBEDDING_DIM, batch_size, HIDDEN_DIM)
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
#inputs = prepare_sequence(training_data[0][0], word_to_ix)
#tag_scores = model(inputs)
countsen = 0
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    #for i in range(len(padded_X)):
    # Step 1. Remember that Pytorch accumulates gradients.
    # We need to clear them out before each instance
    model.zero_grad()

    # Also, we need to clear out the hidden state of the LSTM,
    # detaching it from its history on the last instance.
    model.hidden = model.init_hidden()

    # Step 2. Get our inputs ready for the network, that is, turn them into
    # Variables of word indices.
    targets = torch.tensor(padded_Y, dtype=torch.float)
    # Step 3. Run our forward pass.
    tag_scores = model(xemb)
    print("targets type: ", type(targets))
    print("output type: ", type(tag_scores))
    tag_scores = tag_scores.squeeze()
    print("output shape: ", tag_scores.shape)
    # Step 4. Compute the loss, gradients, and update the parameters by
    #  calling optimizer.step()
    loss = loss_function(tag_scores, targets)
    loss.backward()
    optimizer.step()


'''
# See what the scores are after training
with torch.no_grad():
    inputs = torch.tensor(padded_X[0], dtype=torch.long)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print("The tag scores are: ")
    print(tag_scores)
'''

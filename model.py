import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from dataset import build_vocab, NERDataset, custom_collate
from utils import loss_fn, accuracy

torch.manual_seed(1)

class RNN(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, drop_prob = 0.5):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = 2

        # Embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM model
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=drop_prob, batch_first=True, bidirectional=True, num_layers=self.n_layers)

        # Dropouts
        # self.dropout = nn.Dropout(drop_prob)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence, hidden):

        embeds = self.word_embeddings(sentence)

        # LSTM model
        lstm_out, hidden = self.lstm(embeds)
        output = lstm_out.reshape(-1, lstm_out.shape[2])

        # output = self.dropout(hidden[0][-1])
        tag_space = self.hidden2tag(output)

        # Softmax layer to convert output to probabilities
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.n_layers, batch_size, self.hidden_dim).uniform_()),
                Variable(weight.new(self.n_layers, batch_size, self.hidden_dim).uniform_()))

if __name__ == "__main__":

    EMBEDDING_DIM = 100
    HIDDEN_DIM = 100
    BATCH_SIZE = 256

    vocab = build_vocab('data')
    word_vocab, label_vocab = vocab
    train_dataset = NERDataset('data', vocab, type='/train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, collate_fn=custom_collate, shuffle = True)
    sample_data, sample_target, sample_len = next(iter(train_loader))
    sample_data = sample_data.long()

    model = RNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_vocab), len(label_vocab))
    hidden = model.init_hidden(BATCH_SIZE) 

    with torch.no_grad():
        tag_scores = model(sample_data, hidden)
        print(tag_scores.shape)

    
    loss = loss_fn(tag_scores, sample_target)
    print(loss.item())
    acc, f1 = accuracy(tag_scores, sample_target)
    print(acc, f1)

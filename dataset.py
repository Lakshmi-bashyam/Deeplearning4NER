from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_S = '<s>'
PAD_E = '<e>'
PAD_TAG = '<t>'
UNK_WORD = 'UNK'

def build_vocab(path):
    def update_vocab(vocab, file_path):
        with open(file_path) as f:
            for i, line in enumerate(f):
                vocab.update(line.strip().split(' '))

    def save_vocab_to_txt_file(vocab, txt_path):
        with open(txt_path, "w") as f:
            for token in vocab:
                f.write(token + '\n')
    
    def read_words(vocab):
        vocab_lookup = {}
        for idx, token in enumerate(vocab):
            vocab_lookup[token] = idx
        return vocab_lookup

   
    words = Counter()
    tags = Counter()

    words.update([PAD_S])
    tags.update([PAD_TAG])
    # words.update([PAD_E])
    words.update([UNK_WORD])

    for i in ['/sentences.txt', '/labels.txt']:
        for j in [path +'/train'+i, path +'/test'+i, path +'/val'+i]:
            if 'sentence' in i:
                update_vocab(words, j)
            else:
                update_vocab(tags, j)

    # Save the vocab to a fie
    save_vocab_to_txt_file(words, path + '/words.txt')
    save_vocab_to_txt_file(tags, path + '/tags.txt')

    words = read_words(words)
    tags = read_words(tags)

    return (words, tags)

class NERDataset(Dataset):
    def __init__(self, path, vocab, type='/train'):
        self.data= []
        (word, tag) = vocab
        sentences = open(path + type + '/sentences.txt').read()
        labels = open(path + type + '/labels.txt').read()   
        for sent, label in zip(sentences.split("\n"), labels.split("\n")):
            words = []
            tags = []
            for i, j in zip(sent.split(" "), label.split(" ")):
                words.append(word.get(i, UNK_WORD))
                tags.append(tag.get(j))
            self.data.append([words, tags, len(words)])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate(batch):
    (data, label, lens) = zip(*batch)

    data = [torch.Tensor(line) for line in data]
    label = [torch.Tensor(line) for line in label]

    data_pad = pad_sequence(data, batch_first=True, padding_value=0)
    label_pad = pad_sequence(label, batch_first=True, padding_value=0)

    return data_pad, label_pad, max(lens)

if __name__ == "__main__":

    vocab = build_vocab('data')
    train_dataset = NERDataset('data', vocab, type='/train')
    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=2, collate_fn=custom_collate)

    for word, tag, lens in train_loader:
        print(lens)
        print(len(word[0]))
        break
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import build_vocab, NERDataset, custom_collate
from utils import loss_fn, accuracy
from model import RNN

torch.manual_seed(1)

def train(model, iterator, optimizer):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_f1 = 0.0

    for words, labels, lens in iterator:
        words, labels = words.to(device), labels.to(device)

        optimizer.zero_grad()
        pred = model(words.long(), hidden)
        loss = loss_fn(pred, labels)
        #compute the binary accuracy
        (acc, f1) = accuracy(pred, labels)

        #backpropage the loss and compute the gradients
        loss.backward()       

        #update the weights
        optimizer.step()      
        running_loss += loss.item()
        running_acc += acc
        running_f1 += f1
    
    return running_loss/len(iterator), running_acc/len(iterator), running_f1/len(iterator)

def test(model, iterator):
    running_loss = 0.0
    running_acc = 0.0
    running_f1 = 0.0
    with torch.no_grad():
        for words, labels, lens in iterator:
            words, labels = words.to(device), labels.to(device)

            pred = model(words.long(), hidden)
            loss = loss_fn(pred, labels)
            #compute the binary accuracy
            (acc, f1) = accuracy(pred, labels) 

            running_loss += loss.item()
            running_acc += acc
            running_f1 += f1

    return running_loss/len(iterator), running_acc/len(iterator), running_f1/len(iterator)


if __name__ == '__main__':
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 64
    BATCH_SIZE = 64
    EPOCH = 20
    LR_RATE = 1e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()
    writer.flush()

    # Create train dataloader
    vocab = build_vocab('data')
    word_vocab, label_vocab = vocab
    train_dataset = NERDataset('data', vocab, type='/train')
    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=2, collate_fn=custom_collate, shuffle=True)
    val_dataset = NERDataset('data', vocab, type='/val')
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=2, collate_fn=custom_collate, shuffle=True)

    # Model initialisation
    model = RNN(EMBEDDING_DIM, HIDDEN_DIM, len(word_vocab), len(label_vocab))
    model.to(device)

    # cost function
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE)

    # Define structures for loss, accuracy values 
    training_loss = []
    training_acc = []
    training_f1 = []
    validation_loss = []
    validation_acc = []
    validation_f1 = []
    
    for e in range(EPOCH):
        hidden = model.init_hidden(BATCH_SIZE) 

        # Training and saving the parameters
        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer)

        # Testing on test dataset
        val_loss, val_acc, val_f1 = test(model, val_loader)

        print("Epoch {} - Training loss: {} - Training accuracy: {} Training F1: {}".format(e, train_loss, train_acc, train_f1))
        training_loss.append(train_loss) 
        training_acc.append(train_acc)  
        training_f1.append(train_f1)
        writer.add_scalar('Loss/train', train_loss, e)
        writer.add_scalar('Accuracy/train', train_acc, e)
        writer.add_scalar('F1/train', train_f1, e)


        print("Epoch {} - Validation loss: {} - Validation accuracy: {}, Validation F1: {}".format(e, val_loss, val_acc, val_f1))
        validation_loss.append(val_loss) 
        validation_acc.append(val_acc)
        validation_f1.append(val_f1)
        writer.add_scalar('Loss/test', val_loss, e)
        writer.add_scalar('Accuracy/test', val_acc, e)
        writer.add_scalar('F1/test', val_f1, e)

    PATH = './ner_model.pth'
    torch.save(model.state_dict(), PATH)

    # Test on testing data
    test_dataset = NERDataset('data', vocab, type='/test')
    test_loader = DataLoader(test_dataset, batch_size=1024, num_workers=2, collate_fn=custom_collate, shuffle=True)
    test_loss, test_acc, test_f1 = test(model, test_loader)
    print("Testing loss: {} - Testing accuracy: {}, Testing F1: {}".format(test_loss, test_acc, test_f1))




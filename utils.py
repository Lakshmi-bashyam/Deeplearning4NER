import torch

def accuracy(preds, labels):

    #reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)

    # Get the prediction label
    preds = torch.exp(preds)
    pred_class = preds.argmax(dim=1)

    #mask out 'PAD' tokens
    mask = (labels > 0).float()

    correct_class = ((pred_class == labels) * mask)
    acc = int(correct_class.sum()) / int(torch.sum(mask).item())
    
    return acc


def loss_fn(outputs, labels):
    #reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)  

    #mask out 'PAD' tokens
    mask = (labels > 0).float()

    #the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask).item())

    #pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels.long()]*mask

    #cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs)/num_tokens
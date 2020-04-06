import torch


def to_sentence(index_sentence, vocab):
    n = len(vocab.itos)
    if len(index_sentence.size()) == 1:
        return ' '.join([vocab.itos[w] for w in index_sentence])
    elif index_sentence.size(1) == n:
        _, sentence = torch.softmax(index_sentence, dim=1).max(1)
        return ' '.join([vocab.itos[w] for w in sentence])

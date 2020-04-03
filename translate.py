import argparse
import pickle
from argparse import Namespace
from typing import Generator, List

import torch
from nltk.corpus import wordnet
from torchtext.vocab import Vocab
from tqdm import tqdm

from tools import const
from transformer import load_transformer
from torchtext.data import Dataset, Field

from transformer.translator import Translator


def init() -> Namespace:
    parser = argparse.ArgumentParser(description='translate')
    parser.add_argument('--output', default='pred.txt')
    parser.add_argument('--beam', default=5, type=int)
    parser.add_argument('--cuda', default='cuda')
    parser.add_argument('--data', default='.data/data.pkl')
    parser.add_argument('--max_seq_len', default=100, type=int)
    parser.add_argument('--checkpoint_path', default='checkpoints')

    opt = parser.parse_args()
    opt.device = torch.device('cuda' if opt.cuda else 'cpu')
    return opt


def load_data(opt) -> Dataset:
    data = pickle.load(open(opt.data, 'rb'))
    src: Field = data['src']
    trg: Field = data['trg']

    opt.src_pad_idx = src.vocab.stoi[const.PAD]
    opt.trg_pad_idx = trg.vocab.stoi[const.PAD]
    opt.trg_sos_idx = trg.vocab.stoi[const.SOS]
    opt.trg_eos_idx = trg.vocab.stoi[const.EOS]

    test_loader = Dataset(examples=data['test'], fields={'src': src, 'trg': trg})
    return test_loader


def get_word_or_synonym(vocab: Vocab, word: str, unk_idx: int):
    if word in vocab.stoi:
        return vocab.stoi[word]

    syns = wordnet.synsets(word)
    for s in syns:
        for lemma in s.lemmas():
            if lemma.name() in vocab.stoi:
                print('Synonym 사용:', lemma.name())
                return vocab.stoi[lemma.name()]
    return unk_idx


def iterate_test_data(data_loader: Dataset,
                      device: torch.device) -> Generator[torch.LongTensor, torch.LongTensor, None]:
    src_vocab = data_loader.fields['src'].vocab
    unk_idx = src_vocab.stoi[const.UNK]

    for example in tqdm(data_loader, mininterval=1, desc='Evaluation', leave=False):
        src_seq = [get_word_or_synonym(src_vocab, word, unk_idx) for word in example.src]
        yield torch.LongTensor([src_seq]).to(device)


def main():
    opt = init()

    # Load trained model
    model = load_transformer(opt)

    # Load test dataset
    test_loader = load_data(opt)

    # Load Translator
    translator = Translator(model=load_transformer(opt),
                            beam_size=opt.beam,
                            device=opt.device,
                            max_seq_len=opt.max_seq_len,
                            src_pad_idx=opt.src_pad_idx,
                            trg_pad_idx=opt.trg_pad_idx,
                            trg_sos_idx=opt.trg_sos_idx,
                            trg_eos_idx=opt.trg_eos_idx)

    for src_seq in iterate_test_data(test_loader, device=opt.device):
        translator.translate(src_seq)


if __name__ == '__main__':
    main()

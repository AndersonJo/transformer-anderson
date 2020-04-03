import re

import spacy
from konlpy.tag import Mecab


class Tokenizer(object):
    def __init__(self, lang: str):
        self.nlp = spacy.load(lang)

    def tokenizer(self, sentence):
        sentence = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]


class TokenizerKorea(object):
    def __init__(self):
        self.nlp = Mecab()

    def tokenizer(self, sentence):
        pass

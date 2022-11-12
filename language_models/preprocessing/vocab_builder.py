from collections import Counter
import pickle
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab


class VocabBuilder:
    def __init__(self, tokenizer=get_tokenizer('spacy', language='en_core_web_sm')):
        self.tokenizer = tokenizer

    def build_vocab(self, dataset):
        counter = Counter()
        if dataset.dtype == object:
            for string in dataset:
                counter.update(self.tokenizer(string))
        else:
            raise TypeError("Only objects are allowed")
        return vocab(counter, specials=['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'])

    def save_vocab(self, vocab):
        filename = 'vocab.pkl'
        output = open(filename, 'wb')
        pickle.dump(vocab, output)
        output.close()
        return filename

    def save_corpus(self, dataset):
        raise NotImplementedError()
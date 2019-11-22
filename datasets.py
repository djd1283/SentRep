"""Create dataset of (anchor, pos, neg) examples for training sentence representations from triplet loss. Format SNLI
dataset for fine-tuning."""
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import youtokentome as yttm
import random
import pickle
import nltk
import numpy as np
from nltk import tokenize


PAD_IDX = 0


class GutenbergDataset(Dataset):
    def __init__(self, gutenberg_path=None, bpe_path=None, tmp_path=None, regen_data=True, regen_bpe=True, seed=1234, max_len=40):
        """Dataset of over 3000 english books"""
        super().__init__()
        self.max_len = max_len

        if regen_data:
            # HERE WE REGENERATE BPE AND ALL DATA
            assert gutenberg_path is not None

            # BPE for sentences
            if regen_bpe:
                yttm.BPE.train(data=gutenberg_path, vocab_size=10000, model=bpe_path)

            # Loading model
            self.bpe = yttm.BPE(model=bpe_path)

            print('Vocab size: %s' % self.bpe.vocab_size())

            # run through all lines in dataset
            print('Loading Gutenberg sentences')
            with open(gutenberg_path, 'r') as f:
                books = f.read()

            # concatenate all lines together with spaces in-between
            books_stripped = ' '.join(books.split())  # remove extra whitespace
            #sentences = books_stripped.split('.')

            print('Splitting sentences')
            sentences = tokenize.sent_tokenize(books_stripped)



            # converting all sentences to BPE
            print('Converting sentences to BPE')
            # sentences = [self.bpe.encode(sentence, output_type=yttm.OutputType.SUBWORD, bos=True, eos=True) for sentence in tqdm(sentences)]
            print('Average sentence length: %s' % np.mean([len(sentence.split()) for sentence in sentences]))
            print('Average sentence std: %s' % np.std([len(sentence.split()) for sentence in sentences]))

            # create list of sentence pairs
            print('Creating sentence pairs')
            sentence_pairs = list(zip(sentences[:-1], sentences[1:]))

            print('Shuffling')
            random.seed(seed)
            random.shuffle(sentence_pairs)

            # examples consist of (anchor, positive, negative)
            print('Creating triplet examples (anchor, positive, negative)')
            self.examples = [(sentence_pairs[i][0], sentence_pairs[i][1], sentence_pairs[i + 1][1]) for i in range(len(sentence_pairs) - 1)]

            pickle.dump(self.examples, open(tmp_path, 'wb'))

        else:
            print('Loading examples from tmp file')
            # OTHERWISE WE LOAD DATA FROM SAVE
            self.examples = pickle.load(open(tmp_path, 'rb'))

            # Loading model
            self.bpe = yttm.BPE(model=bpe_path)

    def __len__(self):
        return len(self.examples)

    def format_example(self, next_example):
        """We use NLTK tokenizer"""
        # tokenize to split punctuation
        next_example = [' '.join(nltk.word_tokenize(sentence)) for sentence in next_example]
        # BPE encode
        next_example = [self.bpe.encode(sentence, output_type=yttm.OutputType.ID, bos=True, eos=True) for sentence in
                        next_example]
        # clip to max length
        next_example = [sentence[:self.max_len] for sentence in next_example]
        # pad to max length
        next_example = [sentence + [PAD_IDX] * (self.max_len - len(sentence)) for sentence in next_example]
        # convert to numpy array
        next_example = [np.array(sentence) for sentence in next_example]
        return next_example

    def __getitem__(self, idx):
        next_example = self.examples[idx]
        next_example = self.format_example(next_example)
        return next_example


if __name__ == '__main__':
    regenerate = True
    gutenberg_path = 'data/Gutenberg/txt/Zane Grey___Betty Zane.txt'
    #gutenberg_path = 'data/Gutenberg/txt/all.txt'
    bpe_path = 'data/bpe.model'
    tmp_path = 'data/Gutenberg/processed.pkl'
    ds = GutenbergDataset(gutenberg_path=gutenberg_path, bpe_path=bpe_path, tmp_path=tmp_path, regen_bpe=regenerate,
                          regen_data=regenerate)

    print('Length of dataset: %s' % len(ds))
    print('Example #0: %s' % str(ds[1000]))
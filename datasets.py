"""Create dataset of (anchor, pos, neg) examples for training sentence representations from triplet loss. Format SNLI
dataset for fine-tuning."""
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import youtokentome as yttm
import random
import pickle
import nltk
from nltk import tokenize


class GutenbergDataset(Dataset):
    def __init__(self, gutenberg_path=None, bpe_path=None, tmp_path=None, regenerate=True, seed=1234):
        """Dataset of over 3000 english books"""
        super().__init__()

        if regenerate:
            # HERE WE REGENERATE BPE AND ALL DATA
            assert gutenberg_path is not None

            # BPE for sentences
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

    def tokenize(self, s):
        """We use NLTK tokenizer"""
        result = ' '.join(nltk.word_tokenize(s))
        return result

    def __getitem__(self, idx):
        next_example = self.examples[idx]
        next_example = [self.tokenize(sentence) for sentence in next_example]
        next_example = [self.bpe.encode(sentence, output_type=yttm.OutputType.SUBWORD, bos=True, eos=True) for sentence in next_example]
        return next_example


if __name__ == '__main__':
    regenerate = False
    gutenberg_path = 'data/Gutenberg/txt/Zane Grey___Betty Zane.txt'
    #gutenberg_path = 'data/Gutenberg/txt/all.txt'
    bpe_path = 'data/bpe.model'
    tmp_path = 'data/Gutenberg/processed.pkl'
    ds = GutenbergDataset(gutenberg_path=gutenberg_path, bpe_path=bpe_path, tmp_path=tmp_path, regenerate=regenerate)

    print('Length of dataset: %s' % len(ds))

    print('Example #0: %s' % str(ds[1000]))
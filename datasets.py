"""Create dataset of (anchor, pos, neg) examples for training sentence representations from triplet loss. Format SNLI
dataset for fine-tuning."""
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
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
    def __init__(self, gutenberg_path=None, bpe_path=None, tmp_path=None, regen_data=True, regen_bpe=True,
                 d_vocab=10000, seed=1234, max_len=40, bert=True):
        """Dataset of over 3000 english books"""
        super().__init__()
        self.max_len = max_len
        self.bert = bert

        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        example_path = os.path.join(tmp_path, 'examples.pkl')

        if bert:
            pretrained_weights = 'bert-base-uncased'
            self.wordpiece = BertTokenizer.from_pretrained(pretrained_weights)

        if regen_data:
            # HERE WE REGENERATE BPE AND ALL DATA
            assert gutenberg_path is not None

            # TODO found bug, where BPE is not being trained on tokenized sentences

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

            pickle.dump(self.examples, open(example_path, 'wb'))

            # Loading BPE model
            self.bpe = yttm.BPE(model=bpe_path)

            # BPE for sentences
            if regen_bpe:
                yttm.BPE.train(data=gutenberg_path, vocab_size=d_vocab, model=bpe_path)

        else:
            print('Loading examples from tmp file')
            # OTHERWISE WE LOAD DATA FROM SAVE
            self.examples = pickle.load(open(example_path, 'rb'))

            # Loading model
            self.bpe = yttm.BPE(model=bpe_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        next_example = self.examples[idx]
        if self.bert:
            next_example = [format_sentence_with_bert(sentence, self.wordpiece, self.max_len) for sentence in next_example]
        else:
            next_example = [format_sentence(sentence, self.bpe, self.max_len) for sentence in next_example]
        return next_example


def format_sentence_with_bert(sentence, wordpiece, max_len):
    """Convert sentence to BERT sentence ids using wordpiece."""

    wordpiece_enc = wordpiece.encode(sentence, add_special_tokens=True, max_length=max_len)

    # pad to max length
    wordpiece_enc = wordpiece_enc + [0] * (max_len - len(wordpiece_enc))

    indices = np.array(wordpiece_enc)

    return indices


def format_sentence(sentence, bpe, max_len):
    """We use NLTK tokenizer"""
    # tokenize to split punctuation
    sentence = ' '.join(nltk.word_tokenize(sentence))
    # BPE encode
    sentence = bpe.encode(sentence, output_type=yttm.OutputType.ID, bos=True, eos=True)
    # clip to max length
    sentence = sentence[:max_len]
    # pad to max length
    sentence = sentence + [PAD_IDX] * (max_len - len(sentence))
    # convert to numpy array
    sentence = np.array(sentence)
    return sentence


class SNLIDataset(Dataset):
    def __init__(self, snli_path, tmp_path, regenerate=True, max_len=40, multinli_path=None):
        super().__init__()
        self.max_len = max_len

        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        examples_path = os.path.join(tmp_path, 'examples.pkl')

        pretrained_weights = 'bert-base-uncased'
        self.wordpiece = BertTokenizer.from_pretrained(pretrained_weights)

        if regenerate:
            print('Regenerating SNLI formatted features')
            self.snli_path = snli_path

            data = open(self.snli_path, 'r').read()
            lines = data.split('\n')[1:]  # first line is just header

            # if we choose to include mnli data, add it here
            if multinli_path is not None:
                m_data = open(multinli_path, 'r').read()
                m_lines = m_data.split('\n')[1:]
                lines.extend(m_lines)

            features = [line.split('\t') for line in lines]
            # premise, hypothesis, label for each example
            # import pdb; pdb.set_trace()
            self.examples = [(feature[5], feature[6], feature[0]) for feature in features if len(feature) > 6]

            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)

            pickle.dump(self.examples, open(examples_path, 'wb'))
        else:
            print('Loading formatted SNLI from tmp path')
            self.examples = pickle.load(open(examples_path, 'rb'))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        premise = format_sentence_with_bert(self.examples[idx][0], self.wordpiece, self.max_len)
        hypothesis = format_sentence_with_bert(self.examples[idx][1], self.wordpiece, self.max_len)
        label = self.examples[idx][2]
        label = 0 if label == 'neutral' else 1 if label == 'entailment' else 2  # if label == 'contradiction'

        return premise, hypothesis, label


class WikiTextDataset(Dataset):
    def __init__(self, wiki_path=None, tmp_path=None, regenerate=True, seed=1234, max_len=40, include_right_context=True):
        """Dataset of over 3000 english books"""
        super().__init__()
        self.max_len = max_len
        self.include_right_context = include_right_context

        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        example_path = os.path.join(tmp_path, 'examples.pkl')

        pretrained_weights = 'bert-base-uncased'
        self.wordpiece = BertTokenizer.from_pretrained(pretrained_weights)

        if regenerate:
            # HERE WE REGENERATE BPE AND ALL DATA

            # TODO found bug, where BPE is not being trained on tokenized sentences

            # run through all lines in dataset
            print('Loading Wiki sentences')
            with open(wiki_path, 'r') as f:
                books = f.read()

            # concatenate all lines together with spaces in-between
            books_stripped = ' '.join(books.split())  # remove extra whitespace
            #sentences = books_stripped.split('.')

            print('Splitting sentences')
            sentences = tokenize.sent_tokenize(books_stripped)

            # sentences = [self.bpe.encode(sentence, output_type=yttm.OutputType.SUBWORD, bos=True, eos=True) for sentence in tqdm(sentences)]
            print('Average sentence length: %s' % np.mean([len(sentence.split()) for sentence in sentences]))
            print('Average sentence std: %s' % np.std([len(sentence.split()) for sentence in sentences]))

            # create list of sentence pairs
            print('Creating sentence pairs')
            #sentence_pairs = list(zip(sentences[:-1], sentences[1:]))

            examples = []
            print('Computing sentence outer and inner pairs')
            # here we try to predict positive/negative from anchor
            for i in tqdm(range(len(sentences) - 2)):
                rand_sent = random.choice(sentences)
                # sentence1, sentence2, sentence3, rand sentence
                if include_right_context:
                    examples.append((sentences[i] + ' [SEP] ' + sentences[i+2], sentences[i+1], rand_sent))
                else:
                    # only include left context as anchor
                    examples.append((sentences[i], sentences[i+1], rand_sent))

            print('Shuffling')
            random.seed(seed)
            random.shuffle(examples)

            # examples consist of (anchor, positive, negative)
            print('Creating triplet examples (anchor, positive, negative)')
            self.examples = examples

            pickle.dump(self.examples, open(example_path, 'wb'))

        else:
            print('Loading examples from tmp file')
            # OTHERWISE WE LOAD DATA FROM SAVE
            self.examples = pickle.load(open(example_path, 'rb'))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        next_example = self.examples[idx]
        anchor = format_sentence_with_bert(next_example[0], self.wordpiece, self.max_len * 2)
        pos = format_sentence_with_bert(next_example[1], self.wordpiece, self.max_len)
        neg = format_sentence_with_bert(next_example[2], self.wordpiece, self.max_len)

        return anchor, pos, neg


def main():
    wiki_path = 'data/wikitext-2/wiki.train.tokens'
    wiki_tmp = 'data/wiki_tmp'

    ds = WikiTextDataset(wiki_path=wiki_path, tmp_path=wiki_tmp, regenerate=True)

    for i in range(10):
        print(ds.examples[i])


if __name__ == '__main__':
    main()

    # regenerate = True
    # gutenberg_path = 'data/Gutenberg/txt/Zane Grey___Betty Zane.txt'
    # #gutenberg_path = 'data/Gutenberg/txt/all.txt'
    # bpe_path = 'data/bpe.model'
    # tmp_path = 'data/Gutenberg/processed.pkl'
    # ds = GutenbergDataset(gutenberg_path=gutenberg_path, bpe_path=bpe_path, tmp_path=tmp_path, regen_bpe=regenerate,
    #                       regen_data=regenerate)
    #
    # print('Length of dataset: %s' % len(ds))
    # print('Example #0: %s' % str(ds[1000]))

    # snli_train_path = 'data/snli_1.0/snli_1.0_train.txt'
    # snli_tmp_path = 'data/snli_tmp'
    # ds = SNLIDataset(snli_train_path, snli_tmp_path, regenerate=True)
    # print(len(ds))
    # example = ds[0]
    # print('Premise: %s' % str(example[0]))
    # print('Hypothesis: %s' % str(example[1]))
    # print('Label: %s' % example[2])








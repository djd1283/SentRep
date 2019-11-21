"""Train model (such as LSTM, Transformer, BERT) on triplet loss and Stanford Natural Language Inference (SNLI)."""
from torch.utils.data import DataLoader
from datasets import GutenbergDataset
from tqdm import tqdm


def train():
    print('Training')


if __name__ == '__main__':
    regenerate = False
    gutenberg_path = 'data/Gutenberg/txt/Zane Grey___Betty Zane.txt'
    # gutenberg_path = 'data/Gutenberg/txt/all.txt'
    bpe_path = 'data/bpe.model'
    tmp_path = 'data/Gutenberg/processed.pkl'
    ds = GutenbergDataset(gutenberg_path=gutenberg_path, bpe_path=bpe_path, tmp_path=tmp_path, regenerate=regenerate)

    print('Length of dataset: %s' % len(ds))
    print('Example #0: %s' % str(ds[1000]))

    dl = DataLoader(ds)

    train
















"""Place to store all hyperparameters"""

regenerate = True
small = True

d_hidden = 768
n_epoch = 10
restore = False
d_vocab = 10000
bert = False
if bert:
    print('Using BERT encodings')
    d_vocab = 30000

if small:
    train_path = 'data/Gutenberg/train_small.txt'
    val_path = 'data/Gutenberg/val_small.txt'
    bpe_path = 'data/bpe_small.model'
    train_tmp_path = 'data/Gutenberg/train_small_processed.pkl'
    val_tmp_path = 'data/Gutenberg/val_small_processed.pkl'
    model_path = 'data/smallgrureader.ckpt'
else:
    train_path = 'data/Gutenberg/train.txt'
    val_path = 'data/Gutenberg/val.txt'
    bpe_path = 'data/bpe.model'
    train_tmp_path = 'data/Gutenberg/train_processed.pkl'
    val_tmp_path = 'data/Gutenberg/val_processed.pkl'
    model_path = 'data/grureader.ckpt'

if bert:
    model_path = model_path + '.bert'
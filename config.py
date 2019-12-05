"""Place to store all hyperparameters"""

regenerate = True
small = True
lr = 0.00005
batch_size = 16
d_hidden = 768
n_epoch = 10
restore = False
d_vocab = 10000
max_len = 40
train_bert = True
calc_val_loss_every_n = None
path_to_senteval = '/home/ddonahue/SentEval/data'

model_name = 'bertgrureader'
bert = 'bert' in model_name

if 'bert' in model_name:
    print('Using BERT encodings')
    d_vocab = 30000

model_path = model_name + '.ckpt'

if small:
    train_path = 'data/Gutenberg/train_small.txt'
    val_path = 'data/Gutenberg/val_small.txt'
    bpe_path = 'data/bpe_small.model'
    train_tmp_path = 'data/Gutenberg/train_small_processed.pkl'
    val_tmp_path = 'data/Gutenberg/val_small_processed.pkl'
    model_path = 'small_' + model_path
else:
    train_path = 'data/Gutenberg/train.txt'
    val_path = 'data/Gutenberg/val.txt'
    bpe_path = 'data/bpe.model'
    train_tmp_path = 'data/Gutenberg/train_processed.pkl'
    val_tmp_path = 'data/Gutenberg/val_processed.pkl'

# if either of these is true, replace model with BERT max or mean during evaluation
bert_max = False
bert_mean = False
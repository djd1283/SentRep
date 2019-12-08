"""Place to store all hyperparameters"""

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='bertmean')
parser.add_argument('--regenerate', action='store_true', default=True)
parser.add_argument('--small', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.00002)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--d_hidden', type=int, default=768)
parser.add_argument('--n_epoch', type=int, default=1)
parser.add_argument('--restore', action='store_true', default=False)
parser.add_argument('--d_vocab', type=int, default=10000)
parser.add_argument('--max_len', type=int, default=40)
parser.add_argument('--train_bert', action='store_true', default=False)
parser.add_argument('--calc_val_loss_every_n', default=None)
parser.add_argument('--path_to_senteval', type=str, default='/home/ddonahue/SentEval/data')
parser.add_argument('--pretrained_weights', type=str, default='bert-base-uncased')
parser.add_argument('--data', type=str, default='snli')
parser.add_argument('--snli_train_path', type=str, default='data/snli_1.0/snli_1.0_train.txt')
parser.add_argument('--snli_val_path', type=str, default='data/snli_1.0/snli_1.0_dev.txt')
parser.add_argument('--snli_train_tmp_path', type=str, default='data/snli_train_tmp')
parser.add_argument('--snli_val_tmp_path', type=str, default='data/snli_dev_tmp')
parser.add_argument('--bert_max', action='store_true', default=False)
parser.add_argument('--bert_mean', action='store_true', default=False)
parser.add_argument('--model_path', type=str, default=None)

opt = parser.parse_args()

#
# snli_train_path = 'data/snli_1.0/snli_1.0_train.txt'
# snli_val_path = 'data/snli_1.0/snli_1.0_dev.txt'
# snli_train_tmp_path = 'data/snli_train_tmp'
# snli_val_tmp_path = 'data/snli_dev_tmp'
#
# regenerate = False
# small = False
# lr = 0.00002
# batch_size = 16
# d_hidden = 768
# n_epoch = 1
# restore = False
# d_vocab = 10000
# max_len = 40
# train_bert = True
# calc_val_loss_every_n = None
# path_to_senteval = '/home/ddonahue/SentEval/data'
# pretrained_weights = 'bert-base-uncased'
# data = 'snli'

# model_name = 'bertmean'
opt.bert = 'bert' in opt.model_name

if 'bert' in opt.model_name:
    print('Using BERT encodings')
    opt.d_vocab = 30000

if opt.model_path is None:
    opt.model_path = opt.data + '_' + opt.model_name + '.ckpt'

if opt.small:
    opt.train_path = 'data/Gutenberg/train_small.txt'
    opt.val_path = 'data/Gutenberg/val_small.txt'
    opt.bpe_path = 'data/bpe_small.model'
    opt.train_tmp_path = 'data/Gutenberg/train_small_processed'
    opt.val_tmp_path = 'data/Gutenberg/val_small_processed'
    opt.model_path = 'small_' + opt.model_path
else:
    opt.train_path = 'data/Gutenberg/train.txt'
    opt.val_path = 'data/Gutenberg/val.txt'
    opt.bpe_path = 'data/bpe.model'
    opt.train_tmp_path = 'data/Gutenberg/train_processed'
    opt.val_tmp_path = 'data/Gutenberg/val_processed'

opt.params_senteval = {'task_path': opt.path_to_senteval, 'usepytorch': True, 'kfold': 10, 'batch_size': opt.batch_size,
                       'classifier': {'nhid': 0, 'optim': 'adam', 'batch_size': opt.batch_size, 'tenacity': 5, 'epoch_size': 4}}

#
# # if either of these is true, replace model with BERT max or mean during evaluation
# bert_max = False
# bert_mean = False


# we have all variables in opt, now we export them to locals

# dict_opt = vars(opt)
#
# for option in dict_opt:
#     globals()[option] = dict_opt[option]



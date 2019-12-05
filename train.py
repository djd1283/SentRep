"""Train model (such as LSTM, Transformer, BERT) on triplet loss and Stanford Natural Language Inference (SNLI)."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import GutenbergDataset
from models import GRUReader
from tqdm import tqdm
import wandb
from transformers import BertModel
from config import *
from config import batch_size

wandb.init(project='sent_repr')
device = torch.device('cuda:2')


def triplet_loss(anchor_emb, pos_emb, neg_emb):
    # triplet loss = max(||sa-sp|| - ||sa-sn|| + e, 0) for e = 1

    e = 1  # this is the margin between pos and neg examples

    # shape batch_size
    loss = (anchor_emb - pos_emb).norm(dim=1) - (anchor_emb - neg_emb).norm(dim=1) + e

    zero_loss = torch.zeros_like(loss)

    both_losses = torch.stack([loss, zero_loss], 1)

    max_loss, max_idx = torch.max(both_losses, 1)

    return max_loss.mean()


def calc_val_loss(model, val_ds, bert_model=None):

    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    losses = []
    for batch in tqdm(val_dl):

        with torch.no_grad():
            batch = [d.to(device) for d in batch]
            model.eval()
            anchor_s, pos_s, neg_s = batch

            local_batch_size, s_len = anchor_s.shape
            all_s = torch.stack([anchor_s, pos_s, neg_s], 0)
            all_s = all_s.view(local_batch_size * 3, s_len)

            all_emb = calculate_model_outputs(model, all_s, bert_model=bert_model)

            emb_size = all_emb.shape[-1]

            all_emb = all_emb.view(3, anchor_s.shape[0], emb_size)
            anchor_emb = all_emb[0]
            pos_emb = all_emb[1]
            neg_emb = all_emb[2]

            loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
            losses.append(loss)

    return torch.mean(torch.stack(losses, 0))


def calculate_model_outputs(model, all_s, bert_model=None):
    # here we take wordpiece indices and convert them to embeddings with BERT
    if bert_model is not None:
        with torch.no_grad():
            all_bert_emb, _ = bert_model(all_s)[-2:]
            all_s = all_bert_emb  # we represent sentences using BERT embeddings
    # then we change model to take embeddings as input

    # compute sentence representations for anchor, positive, and negative sentences
    all_emb = model(all_s)

    return all_emb


def train(model, train_ds, val_ds):
    print('Training')

    model.to(device)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if calc_val_loss_every_n is None:
        # calculate validation loss at the end of every epoch
        val_loss_every_n = len(train_dl)
    else:
        val_loss_every_n = calc_val_loss_every_n

    lowest_loss = float('inf')
    start_epoch = 0

    bert_model = None
    if bert:
        pretrained_weights = 'bert-base-uncased'
        bert_model = BertModel.from_pretrained(pretrained_weights)
        bert_model.to(device)
        # bert_model = nn.DataParallel(bert_model)

    if restore:
        print('Loading model from save')
        save_details = torch.load(model_path)
        model.load_state_dict(save_details['state_dict'])
        optimizer.load_state_dict(save_details['optimizer'])
        lowest_loss = save_details['best_loss']
        start_epoch = save_details['epoch']
        print('Lowest loss: %s' % lowest_loss)
        print('Current epoch: %s' % start_epoch)

    #print('Evaluating on validation set before training')
    #val_loss = calc_val_loss(model, val_ds, batch_size=batch_size, bert_model=bert_model)
    #wandb.log({'val_loss': val_loss.item()})

    for epoch_idx in range(start_epoch, n_epoch):
        bar = tqdm(train_dl)
        for batch_idx, batch in enumerate(bar):
            batch = [d.to(device) for d in batch]
            anchor_s, pos_s, neg_s = batch

            local_batch_size, s_len = anchor_s.shape
            all_s = torch.stack([anchor_s, pos_s, neg_s], 0)
            all_s = all_s.view(local_batch_size * 3, s_len)

            model.train()

            all_emb = calculate_model_outputs(model, all_s, bert_model=bert_model)

            emb_size = all_emb.shape[-1]

            all_emb = all_emb.view(3, anchor_s.shape[0], emb_size)
            anchor_emb = all_emb[0]
            pos_emb = all_emb[1]
            neg_emb = all_emb[2]

            loss = triplet_loss(anchor_emb, pos_emb, neg_emb)

            # backpropagate gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({'train_loss': loss.item()})

            if batch_idx % val_loss_every_n == val_loss_every_n - 1:
                val_loss = calc_val_loss(model, val_ds, bert_model=bert_model)
                bar.write('Calculated validation loss: %s' % val_loss.item())
                wandb.log({'val_loss': val_loss.item()})

                # if this is the lowest validation loss we've seen, save model
                if val_loss < lowest_loss:
                    lowest_loss = val_loss

                    save_details = {
                        'epoch': epoch_idx + 1,
                        'state_dict': model.state_dict(),
                        'best_loss': lowest_loss,
                        'optimizer' : optimizer.state_dict(),
                    }

                    print('Saving model')
                    torch.save(save_details, model_path)


if __name__ == '__main__':
    ###### PARAMETERS ##############

    ################################

    train_ds = GutenbergDataset(gutenberg_path=train_path, bpe_path=bpe_path, tmp_path=train_tmp_path, regen_data=regenerate,
                                regen_bpe=regenerate, d_vocab=d_vocab, bert=bert, max_len=max_len)

    val_ds = GutenbergDataset(gutenberg_path=val_path, bpe_path=bpe_path, tmp_path=val_tmp_path, regen_data=regenerate,
                                regen_bpe=False, d_vocab=d_vocab, bert=bert, max_len=max_len)

    if bert:
        d_vocab = len(train_ds.wordpiece.ids_to_tokens)
        print('BERT vocab size:', d_vocab)

    print('Length of train dataset: %s' % len(train_ds))
    print('Length of val dataset: %s' % len(val_ds))
    print('Example #0: %s' % str(train_ds[1000]))

    if bert:
        model = GRUReader(d_hidden=d_hidden, d_in=768)
    else:
        model = GRUReader(d_hidden=d_hidden, d_vocab=d_vocab)

    train(model, train_ds, val_ds)
















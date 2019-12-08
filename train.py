"""Train model (such as LSTM, Transformer, BERT) on triplet loss and Stanford Natural Language Inference (SNLI)."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import GutenbergDataset, SNLIDataset
from models import *
from tqdm import tqdm
import wandb
from transformers import BertModel
from config import *
from eval import evaluate
from utils import select_model, calculate_model_outputs

device = torch.device('cuda:0')


def triplet_loss(anchor_emb, pos_emb, neg_emb):
    # triplet loss = max(||sa-sp|| - ||sa-sn|| + e, 0) for e = 1

    e = 1  # this is the margin between pos and neg examples

    # shape batch_size
    loss = (anchor_emb - pos_emb).norm(dim=1) - (anchor_emb - neg_emb).norm(dim=1) + e

    zero_loss = torch.zeros_like(loss)

    both_losses = torch.stack([loss, zero_loss], 1)

    max_loss, max_idx = torch.max(both_losses, 1)

    return max_loss.mean()


def calc_val_loss(model, val_ds):

    val_dl = DataLoader(val_ds, batch_size=opt.batch_size, shuffle=False)
    ce = nn.CrossEntropyLoss()
    losses = []
    for batch in tqdm(val_dl):

        with torch.no_grad():
            batch = [d.to(device) for d in batch]
            model.eval()

            loss = calculate_loss(model, batch, ce)

            losses.append(loss)

    return torch.mean(torch.stack(losses, 0))


def calculate_loss(model, batch, ce):
    if opt.data == 'gutenberg':
        anchor_s, pos_s, neg_s = batch
        local_batch_size, s_len = anchor_s.shape
        all_s = torch.stack([anchor_s, pos_s, neg_s], 0)
        all_s = all_s.view(local_batch_size * 3, s_len)
        all_emb = calculate_model_outputs(model, all_s)
        emb_size = all_emb.shape[-1]
        all_emb = all_emb.view(3, anchor_s.shape[0], emb_size)
        anchor_emb = all_emb[0]
        pos_emb = all_emb[1]
        neg_emb = all_emb[2]
        loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
    else:
        # TODO train on SNLI
        premise, hypothesis, label = batch
        all_pred = model(premise, hypothesis)

        loss = ce(all_pred, label)
    return loss


def train(model, train_ds, val_ds):
    print('Training')

    model.to(device)

    train_dl = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    ce = nn.CrossEntropyLoss()

    if opt.calc_val_loss_every_n is None:
        # calculate validation loss at the end of every epoch
        val_loss_every_n = len(train_dl)
    else:
        val_loss_every_n = opt.calc_val_loss_every_n

    lowest_loss = float('inf')
    start_epoch = 0

    if opt.restore:
        print('Loading model from save')
        save_details = torch.load(model_path)
        model.load_state_dict(save_details['state_dict'])
        optimizer.load_state_dict(save_details['optimizer'])
        lowest_loss = save_details['best_loss']
        start_epoch = save_details['epoch']
        print('Lowest loss: %s' % lowest_loss)
        print('Current epoch: %s' % start_epoch)

    if opt.data == 'snli':
        model = SNLIClassifierFromModel(model, opt.d_hidden)
        model.to(device)

    #print('Evaluating on validation set before training')
    #val_loss = calc_val_loss(model, val_ds)
    #wandb.log({'val_loss': val_loss.item()})

    for epoch_idx in range(start_epoch, opt.n_epoch):
        bar = tqdm(train_dl)
        for batch_idx, batch in enumerate(bar):
            model.train()
            batch = [d.to(device) for d in batch]

            loss = calculate_loss(model, batch, ce)

            # backpropagate gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({'train_loss': loss.item()})

            if batch_idx % val_loss_every_n == val_loss_every_n - 1:
                val_loss = calc_val_loss(model, val_ds)
                bar.write('Calculated validation loss: %s' % val_loss.item())
                wandb.log({'val_loss': val_loss.item()})

                # if this is the lowest validation loss we've seen, save model
                if val_loss < lowest_loss:
                    lowest_loss = val_loss

                    save_details = {
                        'epoch': epoch_idx + 1,
                        'state_dict': model.state_dict() if opt.data != 'snli' else model.model.state_dict(),
                        'best_loss': lowest_loss,
                        'optimizer' : optimizer.state_dict(),
                    }

                    print('Saving model')
                    torch.save(save_details, opt.model_path)


def main():
    wandb.init(project='sent_repr', config=opt)

    if opt.data == 'gutenberg':
        train_ds = GutenbergDataset(gutenberg_path=opt.train_path, bpe_path=opt.bpe_path, tmp_path=opt.train_tmp_path, regen_data=opt.regenerate,
                                    regen_bpe=opt.regenerate, d_vocab=opt.d_vocab, bert=opt.bert, max_len=opt.max_len)

        val_ds = GutenbergDataset(gutenberg_path=opt.val_path, bpe_path=opt.bpe_path, tmp_path=opt.val_tmp_path, regen_data=opt.regenerate,
                                    regen_bpe=False, d_vocab=opt.d_vocab, bert=opt.bert, max_len=opt.max_len)
    elif opt.data == 'snli':
        train_ds = SNLIDataset(snli_path=opt.snli_train_path, tmp_path=opt.snli_train_tmp_path, regenerate=opt.regenerate, max_len=opt.max_len)

        val_ds = SNLIDataset(snli_path=opt.snli_val_path, tmp_path=opt.snli_val_tmp_path, regenerate=opt.regenerate, max_len=opt.max_len)
    else:
        raise ValueError('Wrong dataset selected')

    if opt.bert:
        opt.d_vocab = len(train_ds.wordpiece.ids_to_tokens)
        print('BERT vocab size:', opt.d_vocab)

    model = select_model(opt.model_name)

    print('Length of train dataset: %s' % len(train_ds))
    print('Length of val dataset: %s' % len(val_ds))
    print('Example #0: %s' % str(train_ds[1000]))

    parameters = [p for p in model.parameters() if p.requires_grad]
    n_parameters = sum(p.numel() for p in parameters)
    print('Number of parameters: %s' % n_parameters)

    train(model, train_ds, val_ds)

    # after we are done training, why not run evaluation?


if __name__ == '__main__':
    main()
    evaluate()  # we put this out here so all previous vars will be garbage collected
















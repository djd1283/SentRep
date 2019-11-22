"""Train model (such as LSTM, Transformer, BERT) on triplet loss and Stanford Natural Language Inference (SNLI)."""
import torch
from torch.utils.data import DataLoader
from datasets import GutenbergDataset
from models import GRUReader
from tqdm import tqdm
import wandb

wandb.init(project='sent_repr')


def triplet_loss(anchor_emb, pos_emb, neg_emb):
    # triplet loss = max(||sa-sp|| - ||sa-sn|| + e, 0) for e = 1

    e = 1  # this is the margin between pos and neg examples

    # shape batch_size
    loss = (anchor_emb - pos_emb).norm(dim=1) - (anchor_emb - neg_emb).norm(dim=1) + e

    zero_loss = torch.zeros_like(loss)

    both_losses = torch.stack([loss, zero_loss], 1)

    max_loss, max_idx = torch.max(both_losses, 1)

    return max_loss.mean()


def calc_val_loss(model, val_ds, batch_size=32):
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    losses = []
    for batch in tqdm(val_dl):

        with torch.no_grad():
            anchor_s, pos_s, neg_s = batch

            # compute sentence representations for anchor, positive, and negative sentences
            anchor_emb = model(anchor_s)
            pos_emb = model(pos_s)
            neg_emb = model(neg_s)

            loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
            losses.append(loss)

    return torch.mean(torch.stack(losses, 0))


def train(model, train_ds, val_ds, model_path, n_epoch=1, lr=0.001, batch_size=32, calc_val_loss_every_n=None, restore=False):
    print('Training')


    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if calc_val_loss_every_n is None:
        # calculate validation loss at the end of every epoch
        calc_val_loss_every_n = len(train_dl)

    lowest_loss = float('inf')
    start_epoch = 0

    if restore:
        print('Loading model from save')
        save_details = torch.load(model_path)
        model.load_state_dict(save_details['state_dict'])
        optimizer.load_state_dict(save_details['optimizer'])
        lowest_loss = save_details['best_loss']
        start_epoch = save_details['epoch']
        print('Lowest loss: %s' % lowest_loss)
        print('Current epoch: %s' % start_epoch)

    for epoch_idx in range(start_epoch, n_epoch):
        bar = tqdm(train_dl)
        for batch_idx, batch in enumerate(bar):
            anchor_s, pos_s, neg_s = batch

            # compute sentence representations for anchor, positive, and negative sentences
            anchor_emb = model(anchor_s)
            pos_emb = model(pos_s)
            neg_emb = model(neg_s)

            loss = triplet_loss(anchor_emb, pos_emb, neg_emb)

            # backpropagate gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({'train_loss': loss.item()})

            if batch_idx % calc_val_loss_every_n == calc_val_loss_every_n - 1:
                val_loss = calc_val_loss(model, val_ds, batch_size=batch_size)
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
    regenerate = False
    train_path = 'data/Gutenberg/train_small.txt'
    val_path = 'data/Gutenberg/val_small.txt'
    # gutenberg_path = 'data/Gutenberg/txt/all.txt'
    bpe_path = 'data/bpe.model'
    train_tmp_path = 'data/Gutenberg/train_small_processed.pkl'
    val_tmp_path = 'data/Gutenberg/val_small_processed.pkl'
    model_path = 'data/grureader.ckpt'
    d_hidden = 300
    n_epoch = 10
    restore = False
    d_vocab = 10000
    ################################

    train_ds = GutenbergDataset(gutenberg_path=train_path, bpe_path=bpe_path, tmp_path=train_tmp_path, regen_data=regenerate,
                          regen_bpe=regenerate, d_vocab=d_vocab)

    val_ds = GutenbergDataset(gutenberg_path=val_path, bpe_path=bpe_path, tmp_path=val_tmp_path, regen_data=regenerate,
                                regen_bpe=False, d_vocab=d_vocab)

    print('Length of train dataset: %s' % len(train_ds))
    print('Length of val dataset: %s' % len(val_ds))
    print('Example #0: %s' % str(train_ds[1000]))

    model = GRUReader(d_hidden=d_hidden, d_vocab=d_vocab)

    train(model, train_ds, val_ds, model_path, n_epoch=n_epoch, restore=restore)
















import os, sys, json, time
from tqdm import tqdm
from pprint import pprint

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from transformer.transformer import Transformer
from transformer.dataset import CNNDMDataset, paired_collate_fn
import transformer.config
import transformer.optimizers

# default parameters, edit here
arg={}
arg["data_folder"] = os.path.abspath("../../train/transformer")
arg["batch_size"] = 64
arg["d_model"] = 512 # "embedding" size, size of (almost) everything
arg["d_inner_hid"] = 2048 # size of internal ffn hidden dim
arg["d_k"] = 64 # size of keys    
arg["d_v"] = 64 # size of values
arg["n_head"] = 8 # number of attention heads   
arg["n_layers"] = 3 # number of complete layers
arg["n_warmup_steps"] = 4000 # number of steps with custom learning rate
arg["dropout"] = 0.1
#parser.add_argument('-embs_share_weight', action='store_true')
#parser.add_argument('-proj_share_weight', action='store_true')
arg["log"] = arg["data_folder"]
arg["save_model"] = arg["data_folder"]
arg["save_mode"] = "all" # or "best"
arg["cuda"] = False # or true
arg["label_smoothing"] = True

# ##################

def prepare_dataloaders(arg):
    train_loader = torch.utils.data.DataLoader(
        CNNDMDataset(arg["data_folder"], "train"),
        num_workers=1,
        batch_size=arg["batch_size"] ,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
         CNNDMDataset(arg["data_folder"], "dev"),
        num_workers=1,
        batch_size=arg["batch_size"],
        collate_fn=paired_collate_fn)
    
    test_loader = torch.utils.data.DataLoader(
         CNNDMDataset(arg["data_folder"], "test"),
        num_workers=1,
        batch_size=arg["batch_size"],
        collate_fn=paired_collate_fn)
        
    return train_loader, valid_loader, test_loader

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(transformer.config.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(transformer.config.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=transformer.config.PAD, reduction='sum')

    return loss   
    
    
def train_epoch(model, train_loader, optimizer, device, smoothing):
    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(train_loader, mininterval=2, desc='  - (Training)   ', leave=False):
        # batch is a tuple of batch_size elements (64 x seq, 64 x pos, 64 y seq, 64 y pos) -> seq are padded ints (same len), pos are the positions of each element
        
        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)        
        gold = tgt_seq[:, 1:] # gold cuts the first <s> from all Ys, is a Tensor
        
        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(transformer.config.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, valid_loader, device):
    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=2, desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(transformer.config.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy
    
def train(model, train_loader, valid_loader, test_loader, optimizer, device, arg):
    
    log_train_file = None
    log_valid_file = None

    if arg["log"]:
        log_train_file = os.path.join(arg["log"], 'train.log')
        log_valid_file = os.path.join(arg["log"], 'valid.log')

        print('[Info] Training performance will be written to file: {} and {}'.format(log_train_file, log_valid_file))
        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    epoch_i = 0
    while True:
        epoch_i += 1
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, train_loader, optimizer, device, smoothing=arg["label_smoothing"])
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, valid_loader, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': arg,
            'epoch': epoch_i}

        if arg["save_model"]:
            if arg["save_mode"] == 'all':
                model_name = arg["save_model"] + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif arg["save_mode"] == 'best':
                model_name = arg["save_model"] + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))    
    
def main():        
    pprint(arg)    
    
    # load dataset
    train_loader, valid_loader, test_loader = prepare_dataloaders(arg)
    print("Data loaded. Instances: {} train / {} dev / {} test".format(len(train_loader), len(valid_loader), len(test_loader)))
    
    # prepare model
    device = torch.device('cuda' if arg["cuda"]==True else 'cpu')
    
    
    #print(len(train_loader.dataset.w2i)) # nice, we can index internal propertied of CNNDMDataset from the loader!
    print()
    transformer_network = Transformer(
        len(train_loader.dataset.w2i), # src_vocab_size,
        len(train_loader.dataset.w2i), # tgt_vocab_size, is equal to src size
        train_loader.dataset.conf["max_sequence_len"], # max_token_seq_len, from the preprocess config
        tgt_emb_prj_weight_sharing=True, # opt.proj_share_weight,
        emb_src_tgt_weight_sharing=True, #opt.embs_share_weight,
        d_k=arg["d_k"],
        d_v=arg["d_v"],
        d_model=arg["d_model"],
        d_word_vec=arg["d_model"], # d_word_vec,
        d_inner=arg["d_inner_hid"],
        n_layers=arg["n_layers"],
        n_head=arg["n_head"],
        dropout=arg["dropout"]).to(device)
    
    print("Transformer model initialized.")
    print()
    
    # train model    
    optimizer = transformer.optimizers.ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer_network.parameters()), # apply only on parameters that require_grad
            betas=(0.9, 0.98), eps=1e-09),
        arg["d_model"], arg["n_warmup_steps"])

    train(transformer_network, train_loader, valid_loader, test_loader, optimizer, device, arg)    
    
if __name__ == '__main__':
    main()
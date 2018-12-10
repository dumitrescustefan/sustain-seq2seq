import os, sys, json
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import transformer.config

# default parameters
arg={}
arg["data"] = 
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
arg["log"] = None
arg["save_model"] = None
arg["save_mode"] = "all" # or "best"
arg["cuda"] = False # or true
arg["label_smoothing"] = True

# ##################

# define MyDataset
class MyDataset(Dataset):
    def __init__(self, root_dir, type = "train"):               
        self.root_dir = root_dir

        self.X = torch.load(open(os.path.join(root_dir,type+"_X.pt")))
        self.y = torch.load(open(os.path.join(root_dir,type+"_y.pt")))
        
        self.conf = json.load(open(os.path.join(root_dir,"preprocess_settings.json")))
        self.w2i = json.load(open(os.path.join(root_dir,"word2index.json")))
        self.i2w = json.load(open(os.path.join(root_dir,"index2word.json")))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):        
        return self.X[idx], self.y[idx]

def prepare_dataloaders(dataset_folder, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        MyDataset(dataset_folder, "train"),
        num_workers=2,
        batch_size=opt.batch_size,
        #collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
         MyDataset(dataset_folder, "dev"),
        num_workers=2,
        batch_size=opt.batch_size)
        #collate_fn=paired_collate_fn,
    
    test_loader = torch.utils.data.DataLoader(
         MyDataset(dataset_folder, "test"),
        num_workers=2,
        batch_size=opt.batch_size)
        #collate_fn=paired_collate_fn,
    return train_loader, valid_loader, test_loader

def train():
    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

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

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy
    
def main():    
    # load dataset
    
    
    
    # prepare model
    
    
    
    # train model    
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()), # apply only on parameters that require_grad
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device ,opt)
    
    
if __name__ == '__main__':
    main()
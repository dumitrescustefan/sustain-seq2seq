import os, json
import numpy as np
import torch
import torch.utils.data

import transformer.config

def paired_collate_fn(insts):
    src_insts, tgt_insts = list(zip(*insts))
    src_insts = collate_fn(src_insts)
    tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, *tgt_insts)

def collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [transformer.config.PAD] * (max_len - len(inst))
        for inst in insts])

    batch_pos = np.array([
        [pos_i+1 if w_i != transformer.config.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq])

    batch_seq = torch.LongTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos
    
    
class CNNDMDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, type = "train"):               
        self.root_dir = root_dir

        self.X = torch.load(os.path.join(root_dir,type+"_X.pt"))
        self.y = torch.load(os.path.join(root_dir,type+"_y.pt"))
        
        self.conf = json.load(open(os.path.join(root_dir,"preprocess_settings.json")))
        self.w2i = json.load(open(os.path.join(root_dir,"word2index.json")))
        self.i2w = json.load(open(os.path.join(root_dir,"index2word.json")))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):        
        return self.X[idx], self.y[idx]

import os, sys, json
import numpy as np
import torch
import torch.utils.data

import transformer.config

import traceback

def paired_collate_fn(insts):
    # insts contains a batch_size number of (x, y) elements    
    src_insts, tgt_insts = list(zip(*insts))
    # now src is a batch_size(=64) array of x0 .. x63, and tgt is y0, x63 , xi is variable length
    # ex: if a = [(1,2), (3,4), (5,6)]
    # then b, c = list(zip(*a)) => b = (1,3,5) and b = (2,4,6)
    
    # src_insts is now a tuple of batch_size Xes (x0, x63) where xi is an instance
    src_insts = collate_fn(src_insts)    
    # now src_insts is a tuple of (64_padded_Xes, 64_position_of_padded_Xes)    
    tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, *tgt_insts)

def collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''
    max_len = max(len(inst) for inst in insts) # determines max size for all examples

    batch_seq = np.array( [ inst + [transformer.config.PAD] * (max_len - len(inst))  for inst in insts ] )
    # batch_seq is now a max_len object padded with zeroes to the right (for all instances)
    batch_pos = np.array([ [pos_i+1 if w_i != transformer.config.PAD else 0 for pos_i, w_i in enumerate(inst)] for inst in batch_seq])
    # batch_pos is an array of [1,2,3, .., n, 0, 0, .. 0] of len (max_len). it marks positions
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

    def intlist2string(self, lst):
        text = [self.i2w[x] for x in lst]
        return " ".join(text)
        
            
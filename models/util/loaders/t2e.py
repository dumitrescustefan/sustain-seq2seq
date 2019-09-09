import os, sys, json, random
sys.path.append("../../..")

from models.util.lookup import Lookup
import numpy as np
import torch
import torch.utils.data
from functools import partial

def loader(data_folder, batch_size, src_lookup, tgt_lookup, min_seq_len_X = 5, max_seq_len_X = 1000, min_seq_len_y = 5, max_seq_len_y = 1000, custom_filename_prefix = ""):
    src_pad_id = src_lookup.convert_tokens_to_ids(src_lookup.pad_token)
    tgt_pad_id = tgt_lookup.convert_tokens_to_ids(tgt_lookup.pad_token)
    
    train_loader = torch.utils.data.DataLoader(
        BiDataset(data_folder, "train", min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y, custom_filename_prefix),
        num_workers=torch.get_num_threads(),
        batch_size=batch_size,
        collate_fn=partial(paired_collate_fn, src_padding_idx = src_pad_id, tgt_padding_idx = tgt_pad_id),
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        BiDataset(data_folder, "dev", min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y, custom_filename_prefix),
        num_workers=torch.get_num_threads(),
        batch_size=batch_size,
        collate_fn=partial(paired_collate_fn, src_padding_idx = src_pad_id, tgt_padding_idx = tgt_pad_id))
    
    test_loader = torch.utils.data.DataLoader(
        BiDataset(data_folder, "test", min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y, custom_filename_prefix),
        num_workers=torch.get_num_threads(),
        batch_size=batch_size,
        collate_fn=partial(paired_collate_fn, src_padding_idx = src_pad_id, tgt_padding_idx = tgt_pad_id))
            
    return train_loader, valid_loader, test_loader    

def paired_collate_fn(insts, src_padding_idx, tgt_padding_idx):
    # insts contains a batch_size number of (x, y) elements    
    src_insts, tgt_insts = list(zip(*insts))
    # now src is a batch_size(=64) array of x0 .. x63, and tgt is y0 .. x63 ; xi is variable length
    # ex: if a = [(1,2), (3,4), (5,6)]
    # then b, c = list(zip(*a)) => b = (1,3,5) and b = (2,4,6)
    
    # src_insts is now a tuple of batch_size Xes (x0, x63) where xi is an instance
    #src_insts, src_lenghts, tgt_insts, tgt_lenghts = length_collate_fn(src_insts, tgt_insts)       
    
    src_max_len = max(len(inst) for inst in src_insts) # determines max size for all examples
    
    src_seq_lengths = torch.tensor(list(map(len, src_insts)), dtype=torch.long)    
    src_seq_tensor = torch.tensor(np.array( [ inst + [src_padding_idx] * (src_max_len - len(inst)) for inst in src_insts ] ), dtype=torch.long)
    src_seq_mask = torch.tensor(np.array( [ [1] * len(inst) + [0] * (src_max_len - len(inst)) for inst in src_insts ] ), dtype=torch.long)
    
    src_seq_lengths, perm_idx = src_seq_lengths.sort(0, descending=True)
    src_seq_tensor = src_seq_tensor[perm_idx]   
    src_seq_mask = src_seq_mask[perm_idx]   
    
    tgt_max_len = max(len(inst) for inst in tgt_insts)
    
    tgt_seq_lengths = torch.tensor(list(map(len, tgt_insts)), dtype=torch.long)    
    tgt_seq_tensor = torch.tensor(np.array( [ inst + [tgt_padding_idx] * (tgt_max_len - len(inst)) for inst in tgt_insts ] ), dtype=torch.long)
    tgt_seq_mask = torch.tensor(np.array( [ [1] * len(inst) + [0] * (tgt_max_len - len(inst)) for inst in tgt_insts ] ), dtype=torch.long)
    
    tgt_seq_lengths = tgt_seq_lengths[perm_idx]
    tgt_seq_tensor = tgt_seq_tensor[perm_idx]      
    tgt_seq_mask = tgt_seq_mask[perm_idx]   
      
    return ((src_seq_tensor, src_seq_lengths, src_seq_mask), (tgt_seq_tensor, tgt_seq_lengths, tgt_seq_mask))    

class BiDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, type, min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y, custom_filename_prefix):  
        self.root_dir = root_dir

        self.X = []
        self.y = []
        
        if os.path.exists(os.path.join(root_dir,custom_filename_prefix+type+"_X.pt")):
            X = torch.load(os.path.join(root_dir,custom_filename_prefix+type+"_X.pt"))
            y = torch.load(os.path.join(root_dir,custom_filename_prefix+type+"_y.pt"))
            
            cut_over_X = 0
            cut_under_X = 0
            cut_over_y = 0
            cut_under_y = 0
            
            # max len
            for (sx, sy) in zip(X,y):
                if len(sx) > max_seq_len_X:
                    cut_over_X += 1
                elif len(sx) < min_seq_len_X+2:                
                    cut_under_X += 1
                elif len(sy) > max_seq_len_y:
                    cut_over_y += 1
                elif len(sy) < min_seq_len_y+2:                
                    cut_under_y += 1
                else:
                    self.X.append(sx)
                    self.y.append(sy)                    

            c = list(zip(self.X, self.y))
            random.shuffle(c)
            self.X, self.y = zip(*c)
            self.X = list(self.X)
            self.y = list(self.y)
                        
            print("Dataset [{}] loaded with {} out of {} ({}%) sequences.".format(type, len(self.X), len(X), float(100.*len(self.X)/len(X)) ) )
            print("\t\t For X, {} are over max_len {} and {} are under min_len {}.".format(cut_over_X, max_seq_len_X, cut_under_X, min_seq_len_X))
            print("\t\t For y, {} are over max_len {} and {} are under min_len {}.".format(cut_over_y, max_seq_len_y, cut_under_y, min_seq_len_y))
            
            assert(len(self.X)==len(self.y))
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):        
        return self.X[idx], self.y[idx]
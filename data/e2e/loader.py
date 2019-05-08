import os, sys, json
import numpy as np
import torch
import torch.utils.data

def loader(data_folder, batch_size, max_seq_len = 100000, min_seq_len = 1):
    train_loader = torch.utils.data.DataLoader(
        E2EDataset(data_folder, "train", max_seq_len, min_seq_len),
        num_workers=1,
        batch_size=batch_size,
        collate_fn=simple_paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        E2EDataset(data_folder, "dev", max_seq_len, min_seq_len),
        num_workers=1,
        batch_size=batch_size,
        collate_fn=simple_paired_collate_fn)
    
    test_loader = torch.utils.data.DataLoader(
        E2EDataset(data_folder, "test", max_seq_len, min_seq_len),
        num_workers=1,
        batch_size=batch_size,
        collate_fn=simple_paired_collate_fn)
        
        
    w2i = json.load(open(data_folder+"/w2i.json","r",encoding="utf-8"))
    i2w = json.load(open(data_folder+"/i2w.json","r",encoding="utf-8"))
        
    return train_loader, valid_loader, test_loader, w2i, i2w, w2i, i2w
    # returns DataLoader, DataLoader, DataLoader, dict, dict

def simple_paired_collate_fn(insts):
    # insts contains a batch_size number of (x, y) elements    
    src_insts, tgt_insts = list(zip(*insts))
    # now src is a batch_size(=64) array of x0 .. x63, and tgt is y0 .. x63 ; xi is variable length
    # ex: if a = [(1,2), (3,4), (5,6)]
    # then b, c = list(zip(*a)) => b = (1,3,5) and b = (2,4,6)
    
    # src_insts is now a tuple of batch_size Xes (x0, x63) where xi is an instance
    src_insts = pad_sequence(src_insts)  #  64_padded_Xes
    tgt_insts = pad_sequence(tgt_insts)
    return (src_insts, tgt_insts)    
    
def pad_sequence(insts):
    ''' Pad the instance to the max seq length in batch '''
    max_len = max(len(inst) for inst in insts) # determines max size for all examples
    # batch_seq is now a max_len object padded with zeroes to the right (for all instances)
    return torch.LongTensor( np.array( [ inst + [0] * (max_len - len(inst)) for inst in insts ] ) )
        
    
class E2EDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, type = "train", max_seq_len = 100000, min_seq_len = 1):                       
        self.X = torch.load(os.path.join(root_dir,type+"_X.pt"))
        self.y = torch.load(os.path.join(root_dir,type+"_y.pt"))        
        assert(len(self.X)==len(self.y))
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):        
        return self.X[idx], self.y[idx]
   
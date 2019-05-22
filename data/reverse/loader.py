import os, sys, json
import numpy as np
import torch
import torch.utils.data

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<PAD>'
UNK_WORD = '<UNK>'
BOS_WORD = '<BOS>'
EOS_WORD = '<EOS>'


def loader(data_folder, batch_size, max_seq_len = 15, min_seq_len = 5, train_count = 5000, dev_count = 20, vocab_size = 25):
    train_loader = torch.utils.data.DataLoader(
        ReverseDataset(data_folder, "train", max_seq_len, min_seq_len, train_count, vocab_size),
        num_workers=1,
        batch_size=batch_size,
        collate_fn=simple_paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        ReverseDataset(data_folder, "dev", max_seq_len, min_seq_len, dev_count, vocab_size),
        num_workers=1,
        batch_size=batch_size,
        collate_fn=simple_paired_collate_fn)
    
    test_loader = torch.utils.data.DataLoader(
        ReverseDataset(data_folder, "test", max_seq_len, min_seq_len, dev_count, vocab_size),
        num_workers=1,
        batch_size=batch_size,
        collate_fn=simple_paired_collate_fn)
        
    return train_loader, valid_loader, test_loader, train_loader.dataset.src_w2i, train_loader.dataset.src_i2w, train_loader.dataset.tgt_w2i, train_loader.dataset.tgt_i2w
    # returns DataLoader, DataLoader, DataLoader, dict, dict

def simple_paired_collate_fn(insts):
    # insts contains a batch_size number of (x, y) elements    
    src_insts, tgt_insts = list(zip(*insts))
    # now src is a batch_size(=64) array of x0 .. x63, and tgt is y0 .. x63 ; xi is variable length
    # ex: if a = [(1,2), (3,4), (5,6)]
    # then b, c = list(zip(*a)) => b = (1,3,5) and b = (2,4,6)
    
    # src_insts is now a tuple of batch_size Xes (x0, x63) where xi is an instance
    src_insts = simple_collate_fn(src_insts)  #  64_padded_Xes
    tgt_insts = simple_collate_fn(tgt_insts)
    return (src_insts, tgt_insts)    
    
def simple_collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''
    max_len = max(len(inst) for inst in insts) # determines max size for all examples
    # batch_seq is now a max_len object padded with zeroes to the right (for all instances)
    return torch.LongTensor( np.array( [ inst + [PAD] * (max_len - len(inst)) for inst in insts ] ) )
        
    
class ReverseDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, type = "train", max_seq_len = 100000, min_seq_len = 1, count = 100, vocab_size = 25):               
        
        self.root_dir = root_dir
        
        # generate 'count' sequences of random length starting with BOS and ending with EOS
        self.X = []
        self.y = []
    
        for i in range(count):
            slen = np.random.randint(min_seq_len, max_seq_len-2)
            seq = np.random.randint(4, vocab_size, slen)
            reversed_seq = seq[::-1]
            self.X.append([2]+seq.tolist()+[3])
            self.y.append([2]+reversed_seq.tolist()+[3])
            
        print("{} dataset has {} instances (min_seq_len={}, max_seq_len={}, vocab_size={})".format(type, len(self.X), min_seq_len, max_seq_len, vocab_size))
        
        assert(len(self.X)==len(self.y))
         
        # generate vocab         
        self.src_w2i = {}
        self.src_i2w = {}
        
        self.src_w2i['<PAD>'] = 0
        self.src_w2i['<UNK>'] = 1
        self.src_w2i['<BOS>'] = 2
        self.src_w2i['<EOS>'] = 3
        self.src_i2w['0'] = '<PAD>'
        self.src_i2w['1'] = '<UNK>'
        self.src_i2w['2'] = '<BOS>'
        self.src_i2w['3'] = '<EOS>'
        
        for i in range(4, vocab_size):
            self.src_w2i[str(i)] = i
            self.src_i2w[str(i)] = str(i)
        
        self.tgt_w2i = self.src_w2i
        self.tgt_i2w = self.src_i2w

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):        
        return self.X[idx], self.y[idx]

   
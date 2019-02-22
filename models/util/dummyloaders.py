import os, sys, json, random
import numpy as np
import torch
import torch.utils.data

import traceback


PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


def prepare_dataloaders(data_folder, batch_size):
    train_loader = torch.utils.data.DataLoader(
        CNNDMDataset(data_folder, "train"),
        num_workers=1,
        batch_size=batch_size,
        collate_fn=simple_paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
         CNNDMDataset(data_folder, "dev"),
        num_workers=1,
        batch_size=batch_size,
        collate_fn=simple_paired_collate_fn)
    
    test_loader = torch.utils.data.DataLoader(
         CNNDMDataset(data_folder, "test"),
        num_workers=1,
        batch_size=batch_size,
        collate_fn=simple_paired_collate_fn)
        
    return train_loader, valid_loader, test_loader, train_loader.dataset.w2i, train_loader.dataset.i2w
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
 
    
class CNNDMDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, type = "train"):               
        self.root_dir = root_dir
        #self.X = torch.load(os.path.join(root_dir,type+"_X.pt"))
        #print(type(self.X))
        
        choices = [1,4,5,6,7,8,9]
        self.X = []
        self.y = []
        
        if type == "train":
            max = 16*300
        else:
            max = 16*5
        
        for i in range(max):
            cnt = random.choice(choices)
            lst = [2]
            for j in range(5,cnt*2+5):
                lst.append(random.choice(choices))
            lst.append(3)    
            self.X.append(lst)
            self.y.append(lst[::-1])            
        #print(len(self.X[0]))
        #self.X = torch.LongTensor(self.X)
        #self.y = torch.tensor(self.y, dtype = torch.long)
        
        self.conf = json.load(open(os.path.join(root_dir,"preprocess_settings.json")))
        #self.w2i = json.load(open(os.path.join(root_dir,"word2index.json")))
        #self.i2w = json.load(open(os.path.join(root_dir,"index2word.json")))
        self.w2i = {"<pad>":0, "<unk>":1, "<bos>":2, "<eos>":3, "a":4, "b":5, "c":6, "d":7, "e":8, "f":9}
        self.i2w = {"0":"<pad>", "1":"<unk>", "2":"<bos>", "3":"<eos>", "4":"a", "5":"b", "6":"c", "7":"d", "8":"e", "9":"f"}
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):        
        return self.X[idx], self.y[idx]

   
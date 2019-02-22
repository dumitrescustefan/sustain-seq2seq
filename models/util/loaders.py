import os, sys, json
import numpy as np
import torch
import torch.utils.data

import traceback


PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<PAD>'
UNK_WORD = '<UNK>'
BOS_WORD = '<BOS>'
EOS_WORD = '<EOS>'


def prepare_dataloaders(data_folder, batch_size, max_seq_len = -1):
    train_loader = torch.utils.data.DataLoader(
        CNNDMDataset(data_folder, "train", max_seq_len),
        num_workers=1,
        batch_size=batch_size,
        collate_fn=simple_paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
         CNNDMDataset(data_folder, "dev", max_seq_len),
        num_workers=1,
        batch_size=batch_size,
        collate_fn=simple_paired_collate_fn)
    
    test_loader = torch.utils.data.DataLoader(
         CNNDMDataset(data_folder, "test", max_seq_len),
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
        
    
    
#this one returns also the position for x and ys
def prepare_dataloaders_with_pos(data_folder, batch_size, max_seq_len):
    train_loader = torch.utils.data.DataLoader(
        CNNDMDataset(data_folder, "train", max_seq_len),
        num_workers=1,
        batch_size=batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
         CNNDMDataset(data_folder, "dev", max_seq_len),
        num_workers=1,
        batch_size=batch_size,
        collate_fn=paired_collate_fn)
    
    test_loader = torch.utils.data.DataLoader(
         CNNDMDataset(data_folder, "test", max_seq_len),
        num_workers=1,
        batch_size=batch_size,
        collate_fn=paired_collate_fn)
        
    return train_loader, valid_loader, test_loader, train_loader.dataset.w2i, train_loader.dataset.i2w
    # returns DataLoader, DataLoader, DataLoader, dict, dict

def paired_collate_fn(insts):
    # insts contains a batch_size number of (x, y) elements    
    src_insts, tgt_insts = list(zip(*insts))
    # now src is a batch_size(=64) array of x0 .. x63, and tgt is y0 .. x63 ; xi is variable length
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

    batch_seq = np.array( [ inst + [PAD] * (max_len - len(inst)) for inst in insts ] )
    # batch_seq is now a max_len object padded with zeroes to the right (for all instances)
    batch_pos = np.array([ [pos_i+1 if w_i != PAD else 0 for pos_i, w_i in enumerate(inst)] for inst in batch_seq])
    # batch_pos is an array of [1,2,3, .., n, 0, 0, .. 0] of len (max_len). it marks positions
    batch_seq = torch.LongTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)
    return batch_seq, batch_pos 
    
class CNNDMDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, type = "train", max_seq_len = -1):               
        self.root_dir = root_dir

        X = torch.load(os.path.join(root_dir,type+"_X.pt"))
        y = torch.load(os.path.join(root_dir,type+"_y.pt"))
        
        self.X = []
        self.y = []
        cnt = 0
        if max_seq_len!=-1:
            for (sx, sy) in zip(X,y):
                if len(sx) <= max_seq_len:
                    #self.X.append([2]*2000) #test max batch size for GPU memory load
                    self.X.append(sx)
                    self.y.append(sy)                    
                    cnt+=1            
        else:
            self.X = X
            self.y = y
        print("With max_seq_len = {} there are {} out of {} ({}%) sequences left in the {} dataset.".format(max_seq_len, len(self.X), len(X), float(100.*len(self.X)/len(X)), type))
        
        assert(len(self.X)==len(self.y))
        
        #print(type(self.X[0]))
        #print("---------------")
        #self.X = [[1,2,3],[2,3,4]]
        
        self.conf = json.load(open(os.path.join(root_dir,"preprocess_settings.json")))
        self.w2i = json.load(open(os.path.join(root_dir,"word2index.json")))
        self.i2w = json.load(open(os.path.join(root_dir,"index2word.json")))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):        
        return self.X[idx], self.y[idx]

def intlist2words(self, ints, index2words):
    text = [self.index2words[x] for x in ints]
    #return " ".join(text)
    return text
            
def words2ints (words, word2index):
    ints = []
    for word in words:
        if arg["lowercase"] == True:
            word = word.lower() 
        if word in word2index:
            ints.append(word2index[word])
        else:
            ints.append(word2index["<unk>"])
    return ints



   
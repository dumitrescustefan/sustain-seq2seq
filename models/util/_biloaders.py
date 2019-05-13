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


def prepare_dataloaders(data_folder, batch_size, max_seq_len = 100000, min_seq_len = 1):
    train_loader = torch.utils.data.DataLoader(
        BiDataset(data_folder, "train", max_seq_len, min_seq_len),
        num_workers=1,
        batch_size=batch_size,
        collate_fn=simple_paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
         BiDataset(data_folder, "dev", max_seq_len, min_seq_len),
        num_workers=1,
        batch_size=batch_size,
        collate_fn=simple_paired_collate_fn)
    
    test_loader = torch.utils.data.DataLoader(
         BiDataset(data_folder, "test", max_seq_len, min_seq_len),
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
        
    
class BiDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, type = "train", max_seq_len = 100000, min_seq_len = 1):               
        self.root_dir = root_dir

        X = torch.load(os.path.join(root_dir,type+"_X.pt"))
        y = torch.load(os.path.join(root_dir,type+"_y.pt"))
        
        self.X = []
        self.y = []
        cnt = 0
        zero_size = 0
        
        # max len
        for (sx, sy) in zip(X,y):
            if len(sx) <= max_seq_len and len(sx) >= min_seq_len+2:
                #self.X.append([2]*2000) #test max batch size for GPU memory load
                self.X.append(sx)
                self.y.append(sy)                    
                cnt+=1            
            else: #statistics
                if len(sx) < min_seq_len+2:
                    zero_size+=1
                    
        print("With max_seq_len = {} there are {} out of {} ({}%) sequences left in the {} dataset (skipped {} with min_seq_len = {}).".format(max_seq_len, len(self.X), len(X), float(100.*len(self.X)/len(X)), type, zero_size, min_seq_len))
        
        assert(len(self.X)==len(self.y))
        
        #print(type(self.X[0]))
        #print("---------------")
        #self.X = [[1,2,3],[2,3,4]]
        
        self.conf = json.load(open(os.path.join(root_dir,"preprocess_settings.json")))
        self.src_w2i = json.load(open(os.path.join(root_dir,"en_word2index.json")))
        self.src_i2w = json.load(open(os.path.join(root_dir,"en_index2word.json")))
        
        self.tgt_w2i = json.load(open(os.path.join(root_dir,"ro_word2index.json")))
        self.tgt_i2w = json.load(open(os.path.join(root_dir,"ro_index2word.json")))

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



   
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

class BiDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, type = "train", max_seq_len = 100000, min_seq_len = 5):               
        self.root_dir = root_dir

        X = torch.load(os.path.join(root_dir,type+"_X.pt"))
        y = torch.load(os.path.join(root_dir,type+"_y.pt"))
        
        self.X = []
        self.y = []
        cnt = 0
        zero_size = 0
        
        # max len
        for (sx, sy) in zip(X,y):
            if len(sx) <= max_seq_len and len(sx) >= min_seq_len:                
                self.X.append(sx)
                self.y.append(sy)                    
                cnt+=1            
            else: #statistics
                if len(sx) < min_seq_len+2:
                    zero_size+=1
                    
        print("With max_seq_len = {} there are {} out of {} ({}%) sequences left in the {} dataset (skipped {} with min_seq_len = {}).".format(max_seq_len, len(self.X), len(X), float(100.*len(self.X)/len(X)), type, zero_size, min_seq_len))
        
        assert(len(self.X)==len(self.y))
        
        # sort descending
        #self.X, self.y = ( list(t) for t in zip(*sorted(zip(self.X, self.y), key=lambda x: len(x[0]), reverse=True ) ) )
        #print("Sorted {} set. Largest X sequence = {}, smallest = {}".format(type, len(self.X[0]), len(self.X[-1])))
        
        # sort ascending
        self.X, self.y = ( list(t) for t in zip(*sorted(zip(self.X, self.y), key=lambda x: len(x[0]) ) ) )
        print("\tSorted {} set. Largest X sequence = {}, smallest = {}".format(type, len(self.X[-1]), len(self.X[0])))
        
        
        #print(type(self.X[0]))
        #print("---------------")
        #self.X = [[1,2,3],[2,3,4]]
        
        self.conf = json.load(open(os.path.join(root_dir,"preprocess_settings.json")))
        self.src_w2i = json.load(open(os.path.join(root_dir,"fr_word2index.json")))
        self.src_i2w = json.load(open(os.path.join(root_dir,"fr_index2word.json")))
        
        self.tgt_w2i = json.load(open(os.path.join(root_dir,"en_word2index.json")))
        self.tgt_i2w = json.load(open(os.path.join(root_dir,"en_index2word.json")))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):        
        return self.X[idx], self.y[idx]
        

class VariableDataLoader():
    def __init__ (self, dataset_object):
        self.dataset = dataset_object
        self.offset = 0
        self.len = len(dataset_object.X)
        
    def get_next_batch(self, batch_size):
        if self.offset >= self.len:
            raise Exception("Attempted to access instances with offset >= length")
        start = self.offset 
        stop = min(self.offset+batch_size, self.len)
        X = self.dataset.X[start:stop].copy()
        y = self.dataset.y[start:stop].copy()
        
        X_max_seq_len = max(len(X[0]), len(X[-1])) # max no matter which way X is sorted
        y_max_seq_len = max(len(inst) for inst in y) # determines max size for all examples in y (not sorted)
        
        X = torch.LongTensor(  [ inst + [0] * (X_max_seq_len - len(inst)) for inst in X ]  )
        y = torch.LongTensor(  [ inst + [0] * (y_max_seq_len - len(inst)) for inst in y ]  )
        
        return (X, y)    
    
        
def loader(data_folder, batch_size, max_seq_len = 100000, min_seq_len = 5):        
    train_dataset = BiDataset(data_folder, "train", max_seq_len, min_seq_len)
    dev_dataset = BiDataset(data_folder, "dev", max_seq_len, min_seq_len)
    test_dataset = BiDataset(data_folder, "test", max_seq_len, min_seq_len)
    
    train_dataloader = VariableDataLoader(train_dataset)
    return VariableDataLoader(train_dataset), VariableDataLoader(dev_dataset), VariableDataLoader(test_dataset), train_dataset.src_w2i, train_dataset.src_i2w, train_dataset.tgt_w2i, train_dataset.tgt_i2w
     
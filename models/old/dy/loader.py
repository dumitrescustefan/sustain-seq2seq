import os, sys, json, random
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


def loader(data_folder, batch_size, src_w2i, src_i2w, tgt_w2i, tgt_i2w, min_seq_len_X = 5, max_seq_len_X = 1000, min_seq_len_y = 5, max_seq_len_y = 1000):
    train_loader = BiDataset(data_folder, "train", src_w2i, src_i2w, tgt_w2i, tgt_i2w, min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y)
        
    valid_loader = BiDataset(data_folder, "dev", src_w2i, src_i2w, tgt_w2i, tgt_i2w, min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y)
        
    test_loader = BiDataset(data_folder, "test", src_w2i, src_i2w, tgt_w2i, tgt_i2w, min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y)
    
    return train_loader, valid_loader, test_loader, train_loader.src_w2i, train_loader.src_i2w, train_loader.tgt_w2i, train_loader.tgt_i2w
    # returns DataLoader, DataLoader, DataLoader, dict, dict

    
class BiDataset():
    def __init__(self, root_dir, type, src_w2i, src_i2w, tgt_w2i, tgt_i2w, min_seq_len_X = 5, max_seq_len_X = 1000, min_seq_len_y = 5, max_seq_len_y = 1000):  
        self.root_dir = root_dir

        self.X = []
        self.y = []
        
        if os.path.exists(os.path.join(root_dir,type+"_X.pt")):
            X = torch.load(os.path.join(root_dir,type+"_X.pt"))
            y = torch.load(os.path.join(root_dir,type+"_y.pt"))
            
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
        
        if os.path.exists(os.path.join(root_dir,"preprocess_settings.json")):
            self.conf = json.load(open(os.path.join(root_dir,"preprocess_settings.json")))
        else:
            self.conf = None
        self.src_w2i = json.load(open(os.path.join(root_dir, src_w2i)))
        self.src_i2w = json.load(open(os.path.join(root_dir, src_i2w)))
        
        self.tgt_w2i = json.load(open(os.path.join(root_dir, tgt_w2i)))
        self.tgt_i2w = json.load(open(os.path.join(root_dir, tgt_i2w)))
    
    def reset (self):
        self.i = 0
        
    def __next__(self):
        if self.i < len(self.X)-1:
            self.i+=1
            return self.X[self.i], self.y[self.i]
        raise StopIteration()
    
    def __iter__(self):
        return self
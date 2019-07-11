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
    train_loader = torch.utils.data.DataLoader(
        BiDataset(data_folder, "train", src_w2i, src_i2w, tgt_w2i, tgt_i2w, min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y),
        num_workers=torch.get_num_threads(),
        batch_size=batch_size,
        collate_fn=simple_paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        BiDataset(data_folder, "dev", src_w2i, src_i2w, tgt_w2i, tgt_i2w, min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y),
        num_workers=torch.get_num_threads(),
        batch_size=batch_size,
        collate_fn=simple_paired_collate_fn)
    
    test_loader = torch.utils.data.DataLoader(
        BiDataset(data_folder, "test", src_w2i, src_i2w, tgt_w2i, tgt_i2w, min_seq_len_X, max_seq_len_X, min_seq_len_y, max_seq_len_y),
        num_workers=torch.get_num_threads(),
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
    return torch.LongTensor( np.array( [ inst + [0] * (max_len - len(inst)) for inst in insts ] ) )
        
    
class BiDataset(torch.utils.data.Dataset):
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

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):        
        return self.X[idx], self.y[idx]
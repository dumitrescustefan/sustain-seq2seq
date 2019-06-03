import os, sys, json
import numpy as np
import torch
import torch.utils.data
from sklearn.linear_model import LogisticRegression
from scipy import stats

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<PAD>'
UNK_WORD = '<UNK>'
BOS_WORD = '<BOS>'
EOS_WORD = '<EOS>'

class EstimatedDataLoader(): # full data loader (e.g. train_loader)
    def __init__(self, dataset_object):
        self.estimation_X = []        
        self.estimation_y = []
        self.estimator = LogisticRegression(solver='lbfgs', multi_class='multinomial')
        self.estimator_trained = False
        self.X = dataset_object.X
        self.y = dataset_object.y
        self.offset = 0
        self.len = len(X)
        self.shuffle()
        
    
    def shuffle(self):
        c = list(zip(self.X, self.y))
        random.shuffle(c)
        self.X, self.y = zip(*c)
        
    def get_next_batch(self, batch_size, force=False):
        if self.offset >= self.len:
            raise Exception("Attempted to access instances with offset >= length")
        if self.estimator_trained:
            batch_size = self.estimator.predict 
            pass

        if 
            
        start = self.offset 
        stop = min(self.offset+batch_size, self.len)
        X = self.X[start:stop].copy()
        y = self.y[start:stop].copy()
        
        X_tensor, y_tensor = self._pad(X, y)  
        #estimated_batch_size = batch_size
        return X_tensor, y_tensor, X, y#, estimated_batch_size

    def set_estimation_data(self, X_list, y_list, success): # success is 0 or 1
        self.estimation_X.append(self._get_stats_for_batch(X_list, y_list)) # array of features
        self.estimation_y.append(success) # batch_size 

    def fit_estimator (self):
        print("Fitting estimator...")
        estimator.fit(self.estimation_X, self.estimation_y)
        print("Done fitting estimator.")

    def _get_stats_for_batch(self, X_list, y_list): # X and y are lists not tensors.
        # convert to a list of lenghts
        X = [len(elem) for elem in X_list]
        y = [len(elem) for elem in y_list]        
        features = []
        
        # get features for X 
        ds = stats.describe(X)        
        cnt = ds[0]
        min = ds[1][0]
        max = ds[1][1]
        mean = ds[2]
        variance = ds[3]
        skewness = ds[4]
        kurtosis = ds[5]        
        i_0 = vec[0]
        i_25 = vec[int(cnt/4)]
        i_50 = vec[int(cnt/2)]
        i_75 = vec[int(3*cnt/4)]
        i_100 = vec[-1]
        features += [cnt, min, max, mean, variance, skewness, kurtosis, i_0, i_25, i_50, i_75, i_100]
        
        # get features for y 
        ds = stats.describe(y)        
        cnt = ds[0]
        min = ds[1][0]
        max = ds[1][1]
        mean = ds[2]
        variance = ds[3]
        skewness = ds[4]
        kurtosis = ds[5]        
        i_0 = vec[0]
        i_25 = vec[int(cnt/4)]
        i_50 = vec[int(cnt/2)]
        i_75 = vec[int(3*cnt/4)]
        i_100 = vec[-1]
        features += [cnt, min, max, mean, variance, skewness, kurtosis, i_0, i_25, i_50, i_75, i_100]

        return features
        
pred = estimator.predict([[2,4.2]])
print(pred[0])
        self.estimator.fit(X, y)
        
    def _pad (self, X, y):
        X_max_seq_len = max(len(inst) for inst in X) 
        y_max_seq_len = max(len(inst) for inst in y) 
        
        X = torch.LongTensor(  [ inst + [0] * (X_max_seq_len - len(inst)) for inst in X ]  )
        y = torch.LongTensor(  [ inst + [0] * (y_max_seq_len - len(inst)) for inst in y ]  )
        
        return X, y
        

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
        #self.X, self.y = ( list(t) for t in zip(*sorted(zip(self.X, self.y), key=lambda x: len(x[0]) ) ) )
        #print("\tSorted {} set. Largest X sequence = {}, smallest = {}".format(type, len(self.X[-1]), len(self.X[0])))
        
        self.conf = json.load(open(os.path.join(root_dir,"preprocess_settings.json")))
        self.src_w2i = json.load(open(os.path.join(root_dir,"fr_word2index.json")))
        self.src_i2w = json.load(open(os.path.join(root_dir,"fr_index2word.json")))
        
        self.tgt_w2i = json.load(open(os.path.join(root_dir,"en_word2index.json")))
        self.tgt_i2w = json.load(open(os.path.join(root_dir,"en_index2word.json")))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):        
        return self.X[idx], self.y[idx]
        

def loader(data_folder, batch_size, max_seq_len = 100000, min_seq_len = 5):        
    train_dataset = BiDataset(data_folder, "train", max_seq_len, min_seq_len)
    dev_dataset = BiDataset(data_folder, "dev", max_seq_len, min_seq_len)
    test_dataset = BiDataset(data_folder, "test", max_seq_len, min_seq_len)
    
    train_dataloader = EstimatedDataLoader(train_dataset)
    return EstimatedDataLoader(train_dataset), EstimatedDataLoader(dev_dataset), EstimatedDataLoader(test_dataset), train_dataset.src_w2i, train_dataset.src_i2w, train_dataset.tgt_w2i, train_dataset.tgt_i2w
     
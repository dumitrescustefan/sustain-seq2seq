"""
This script creates a vocabulary based on the input folder.

Set input parameters below:
"""
import os

import os, sys, json, glob, collections
from tqdm import tqdm
from itertools import dropwhile
import torch


# create output folder
if not os.path.exists("ready"):
    os.makedirs("ready")

# create word2index and index2word     
print("Creating word2index and index2word ...")
X_word2index = {}
X_index2word = {}
y_word2index = {}
y_index2word = {}

index = -1
with open("bpe_models/cmudict.62.bpe.X.vocab","r",encoding="utf8") as f:    
    for line in f:
        index+=1
        word = line.split("\t")[0]
        X_word2index[word] = index
        X_index2word[str(index)] = word


# generate w2i/i2w for y
with open(os.path.join("raw","cmudict-0.7b.symbols")) as f:
    content = f.readlines()
symbols = [x.strip() for x in content]

for index, word in enumerate(symbols):
    y_word2index[word] = index+4
    y_index2word[str(index+4)] = word
        
# just to be safe, overwrite special markers
y_word2index['<PAD>'] = 0
y_word2index['<UNK>'] = 1
y_word2index['<BOS>'] = 2
y_word2index['<EOS>'] = 3
y_index2word['0']='<PAD>'
y_index2word['1']='<UNK>'
y_index2word['2']='<BOS>'
y_index2word['3']='<EOS>'

json.dump(y_word2index, open(os.path.join("ready", "y_word2index.json"),"w",encoding="utf-8"), sort_keys=True)
json.dump(y_index2word, open(os.path.join("ready", "y_index2word.json"),"w",encoding="utf-8"), sort_keys=True)

        
# just to be safe, overwrite special markers
X_word2index['<PAD>'] = 0
X_word2index['<UNK>'] = 1
X_word2index['<BOS>'] = 2
X_word2index['<EOS>'] = 3
X_index2word['0']='<PAD>'
X_index2word['1']='<UNK>'
X_index2word['2']='<BOS>'
X_index2word['3']='<EOS>'

json.dump(X_word2index, open(os.path.join("ready", "X_word2index.json"),"w",encoding="utf-8"), sort_keys=True)
json.dump(X_index2word, open(os.path.join("ready", "X_index2word.json"),"w",encoding="utf-8"), sort_keys=True)


# process files
import random
import sentencepiece as spm
sp_en = spm.SentencePieceProcessor()
sp_en.Load("bpe_models/cmudict.62.bpe.X.model")

print("Creating train dev and test files ...") 
   
train_X = []
train_y = []
dev_X = []
dev_y = []
test_X = []
test_y = []   
total_len = 0

with open("raw/y.txt", "r", encoding="utf8") as f:
    fr_lines = f.readlines()
    fr_lines = [x.strip() for x in fr_lines]
with open("raw/X.txt", "r", encoding="utf8") as f:
    en_lines = f.readlines()
    en_lines = [x.strip() for x in en_lines]

cnt = 0    
for y_line, X_line in zip(fr_lines, en_lines):
    X_line_sp = sp_en.EncodeAsIds(X_line)
    y_line_sp = []
    for symbol in y_line.split():
        y_line_sp.append(y_word2index[symbol])
    #print(X_line_sp)
    #print(y_line_sp)
    #print()
    x = [X_word2index["<BOS>"]] + X_line_sp + [X_word2index["<EOS>"]]
    y = [y_word2index["<BOS>"]] + y_line_sp + [y_word2index["<EOS>"]]

    # select train dev or test
    if cnt%10!=0:    
        train_X.append(x)
        train_y.append(y)
    else:
        dev_X.append(x)
        dev_y.append(y)
    cnt += 1
    
# save settings
#json.dump(arg, open(os.path.join(arg["output_folder"],"preprocess_settings.json"),"w",encoding="utf-8"), indent=4, sort_keys=True)

# save train dev test
import torch
torch.save(train_X,os.path.join("ready","train_X.pt"))
torch.save(train_y,os.path.join("ready","train_y.pt"))
torch.save(dev_X,os.path.join("ready","dev_X.pt"))
torch.save(dev_y,os.path.join("ready","dev_y.pt"))

print("\n\nTrain has {} examples, dev has {}, and test has {}".format(len(train_X),len(dev_X), len(test_X)))
print("\n\nDone.")


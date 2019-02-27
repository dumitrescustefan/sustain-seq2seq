"""
This script creates a vocabulary based on the input folder.

Set input parameters below:
"""
import os

arg = {}
#arg["fresh_start"] = False # set to True to overwrite everything. This will take a while.
arg["bpe_model_ro"] = "setimes.8K.bpe.ro.model"
arg["bpe_model_en"] = "setimes.8K.bpe.en.model"
arg["output_folder"] = os.path.abspath("ready/setimes.8K.bpe") # where to store the vocab dict and indexes
arg["bpe_model_vocab_ro"] = os.path.abspath("bpe_models/"+arg["bpe_model_ro"]).replace(".model",".vocab")
arg["bpe_model_vocab_en"] = os.path.abspath("bpe_models/"+arg["bpe_model_en"]).replace(".model",".vocab")
arg["bpe_model_model_ro"] = os.path.abspath("bpe_models/"+arg["bpe_model_ro"])
arg["bpe_model_model_en"] = os.path.abspath("bpe_models/"+arg["bpe_model_en"])
arg["input_folder"] = os.path.abspath("raw") 
arg["validation_fraction"] = 0.03 # fraction to use as validation
arg["test_fraction"] = 0.05 # fraction to test on
#arg["full_data_fraction"] = 1. # what fraction from all avaliable data to use (1.0 if you want full dataset)
arg["reverse_x"] = False
# ######################################

import os, sys, json, glob, collections
from tqdm import tqdm
from itertools import dropwhile
import torch

print("Parameters: ")
print(arg)

# create output folder
if not os.path.exists(arg["output_folder"]):
    os.makedirs(arg["output_folder"])

# create word2index and index2word     
print("Creating word2index and index2word ...")
ro_word2index = {}
ro_index2word = {}
en_word2index = {}
en_index2word = {}

index = -1

with open(arg["bpe_model_vocab_ro"],"r",encoding="utf8") as f:    
    for line in f:
        index+=1
        word = line.split("\t")[0]
        ro_word2index[word] = index
        ro_index2word[str(index)] = word

index = -1        
with open(arg["bpe_model_vocab_en"],"r",encoding="utf8") as f:    
    for line in f:
        index+=1
        word = line.split("\t")[0]
        en_word2index[word] = index
        en_index2word[str(index)] = word
        
        
# just to be safe, overwrite special markers
ro_word2index['<PAD>'] = 0
ro_word2index['<UNK>'] = 1
ro_word2index['<BOS>'] = 2
ro_word2index['<EOS>'] = 3
ro_index2word['0']='<PAD>'
ro_index2word['1']='<UNK>'
ro_index2word['2']='<BOS>'
ro_index2word['3']='<EOS>'
en_word2index['<PAD>'] = 0
en_word2index['<UNK>'] = 1
en_word2index['<BOS>'] = 2
en_word2index['<EOS>'] = 3
en_index2word['0']='<PAD>'
en_index2word['1']='<UNK>'
en_index2word['2']='<BOS>'
en_index2word['3']='<EOS>'

json.dump(ro_word2index, open(os.path.join(arg["output_folder"],"ro_word2index.json"),"w",encoding="utf-8"), sort_keys=True)
json.dump(ro_index2word, open(os.path.join(arg["output_folder"],"ro_index2word.json"),"w",encoding="utf-8"), sort_keys=True)

json.dump(en_word2index, open(os.path.join(arg["output_folder"],"en_word2index.json"),"w",encoding="utf-8"), sort_keys=True)
json.dump(en_index2word, open(os.path.join(arg["output_folder"],"en_index2word.json"),"w",encoding="utf-8"), sort_keys=True)


# process files
import random
import sentencepiece as spm
sp_ro = spm.SentencePieceProcessor()
sp_ro.Load(arg["bpe_model_model_ro"])
sp_en = spm.SentencePieceProcessor()
sp_en.Load(arg["bpe_model_model_en"])

print("Creating train dev and test files ...") 
   
train_X = []
train_y = []
dev_X = []
dev_y = []
test_X = []
test_y = []   
total_len = 0

with open(arg["input_folder"]+"/SETIMES.en-ro.ro", "r", encoding="utf8") as f:
    ro_lines = f.readlines()
    ro_lines = [x.strip() for x in ro_lines]
with open(arg["input_folder"]+"/SETIMES.en-ro.en", "r", encoding="utf8") as f:
    en_lines = f.readlines()
    en_lines = [x.strip() for x in en_lines]
    
for r, e in zip(ro_lines, en_lines):
    ri = sp_ro.EncodeAsIds(r)
    re = sp_en.EncodeAsIds(e)
    
    if arg["reverse_x"]:
        re = re[::-1]

    y = [ro_word2index["<BOS>"]] + ri + [ro_word2index["<EOS>"]]
    x = [ro_word2index["<BOS>"]] + re + [ro_word2index["<EOS>"]]

    total_len += len(re)
    # select train dev or test
    train_fraction = 1.0 - arg["validation_fraction"] - arg["test_fraction"]
    dev_fraction = 1.0 - arg["test_fraction"]
    rand = random.random()
    if rand<train_fraction:
        train_X.append(x)
        train_y.append(y)
    elif rand<dev_fraction:
        dev_X.append(x)
        dev_y.append(y)
    else:
        test_X.append(x)
        test_y.append(y)
    
# save settings
json.dump(arg, open(os.path.join(arg["output_folder"],"preprocess_settings.json"),"w",encoding="utf-8"), indent=4, sort_keys=True)

# save train dev test
import torch
torch.save(train_X,os.path.join(arg["output_folder"],"train_X.pt"))
torch.save(train_y,os.path.join(arg["output_folder"],"train_y.pt"))
torch.save(dev_X,os.path.join(arg["output_folder"],"dev_X.pt"))
torch.save(dev_y,os.path.join(arg["output_folder"],"dev_y.pt"))
torch.save(test_X,os.path.join(arg["output_folder"],"test_X.pt"))
torch.save(test_y,os.path.join(arg["output_folder"],"test_y.pt"))
print("\n\nTrain has {} examples, dev has {}, and test has {}".format(len(train_X),len(dev_X), len(test_X)))
print("\n\nDone.")


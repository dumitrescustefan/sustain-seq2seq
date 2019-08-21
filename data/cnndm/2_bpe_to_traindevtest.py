"""
This script creates a vocabulary based on the input folder.

Set input parameters below:
"""
import os

arg = {}
#arg["fresh_start"] = False # set to True to overwrite everything. This will take a while.
arg["bpe_model"] = "cnndm.8K.bpe.model"
arg["bpe_model_vocab"] = os.path.abspath("bpe/bpe_models/"+arg["bpe_model"]).replace(".model",".vocab")
arg["input_folder"] = os.path.abspath("bpe/bpe_processed/"+arg["bpe_model"]) # where the cnn and dm folders contain the processed jsons
arg["output_folder"] = os.path.abspath("bpe/ready/"+arg["bpe_model"]) # where to store the vocab dict and indexes
arg["validation_fraction"] = 0.02 # fraction to use as validation
arg["test_fraction"] = 0.02 # fraction to test on
arg["full_data_fraction"] = .2 # what fraction from all avaliable data to use (1.0 if you want full dataset)
arg["reverse_x"] = False

arg["x_field"] = "x"
arg["y_field"] = "y"

arg["keep_max_y"] = 1 # how many sentences to keep from y (to keep all set y > 5)
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

# list all input files
input_files = glob.glob(os.path.join(arg["input_folder"],"*.json"))
print("Found a total of "+str(len(input_files))+" input json files.")

# create word2index and index2word     
print("Creating word2index and index2word ...")
word2index = {}
index2word = {}

index = -1

with open(arg["bpe_model_vocab"],"r",encoding="utf8") as f:    
    for line in f:
        index+=1
        word = line.split("\t")[0]
        word2index[word] = index
        index2word[str(index)] = word

# just to be safe, overwrite special markers
word2index['<PAD>'] = 0
word2index['<UNK>'] = 1
word2index['<BOS>'] = 2
word2index['<EOS>'] = 3
index2word['0']='<PAD>'
index2word['1']='<UNK>'
index2word['2']='<BOS>'
index2word['3']='<EOS>'

json.dump(word2index, open(os.path.join(arg["output_folder"],"word2index.json"),"w",encoding="utf-8"), sort_keys=True)
json.dump(index2word, open(os.path.join(arg["output_folder"],"index2word.json"),"w",encoding="utf-8"), sort_keys=True)

# create train dev and test files
print("Creating train dev and test files ...") 
import random
random.shuffle(input_files)
if arg["full_data_fraction"]<1.0:
    cutoff = int(len(input_files)*arg["full_data_fraction"])
    print("From the initial {} files, we'll use only {}".format(len(input_files), cutoff))
    input_files = input_files[:cutoff]
   
train_X = []
train_y = []
dev_X = []
dev_y = []
test_X = []
test_y = []   
unks = 0   
total_len = 0
for input_file in tqdm(input_files, unit='json files', ncols=120, total=len(input_files)):    
    js_array = json.load(open(input_file,"r"))
    for article in tqdm(js_array, unit='articles', ncols=120, total=len(js_array), desc = "    "):  
        # process x
        x = [word2index["<BOS>"]]
        for sentence in article[arg["x_field"]]:            
            """if len(x)+len(sentence) < arg["max_sequence_len"]-1:
                x+=words2ints(sentence)
            else:
                break
            """
            if arg["reverse_x"]:
                x+=sentence[::-1]
            else:
                x+=sentence            
        x+= [word2index["<EOS>"]]
            
        # process y
        y = [word2index["<BOS>"]]
        for index, sentence in enumerate(article[arg["y_field"]]):            
            if index>=arg["keep_max_y"]:
                break            
            y+=sentence
        y+= [word2index["<EOS>"]]
        
        # compute unk rate
        for i in range(len(x)):
            if x[i]==1:
                unks+=1                
        for i in range(len(y)):
            if y[i]==1:
                unks+=1                
        total_len += len(x) + len(y)
        #print(y)
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
unk_rate = 100*unks/float(total_len)
print("\n\nTrain has {} examples, dev has {}, and test has {}, unk rate {} %".format(len(train_X),len(dev_X), len(test_X), unk_rate))
print("\n\nDone.")


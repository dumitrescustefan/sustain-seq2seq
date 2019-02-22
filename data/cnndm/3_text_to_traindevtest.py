"""
This script creates a vocabulary based on the input folder.

Set input parameters below:
"""
import os

arg = {}
arg["fresh_start"] = False # set to True to overwrite everything. This will take a while.

arg["input_folder"] = os.path.abspath("../../data/processed") # where the cnn and dm folders contain the processed jsons
arg["output_folder"] = os.path.abspath("../../train/transformer") # where to store the vocab dict and indexes
arg["lowercase"] = True # whether to lowercase or not
arg["max_vocab_size"] = 50000 # maximum number of words in the vocab
arg["max_sequence_len"] = 400 # max length of an instance
arg["validation_fraction"] = 0.05 # fraction to use as validation
arg["test_fraction"] = 0.05 # fraction to test on
arg["full_data_fraction"] = 0.1 # what fraction from all avaliable data to use (1.0 if you want full dataset)

arg["x_field"] = "x_tokenized_original_sentences"
arg["y_field"] = "y_tokenized_original_sentences"

arg["keep_max_y"] = 1 # how many sentences to keep from y (to keep all set y > 5)
# ######################################


def words2ints (words):
    ints = []
    for word in words:
        if arg["lowercase"] == True:
            word = word.lower() 
        if word in word2index:
            ints.append(word2index[word])
        else:
            ints.append(word2index["<unk>"])
    return ints

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
input_files = glob.glob(os.path.join(arg["input_folder"],"cnn","*.json"))
input_files += glob.glob(os.path.join(arg["input_folder"],"dm","*.json"))
print("Found a total of "+str(len(input_files))+" input json files.")

# first step, count words and discard them based lower than the freq count
print("Counting word frequency ...")
frequency_counter_file = os.path.join(arg["output_folder"],"_frequency_counter.json")
if not os.path.exists(frequency_counter_file) or arg["fresh_start"] == True:
    frequency_counter = collections.Counter()
    for input_file in tqdm(input_files, unit='json files', ncols=120, total=len(input_files)):    
        js_array = json.load(open(input_file,"r"))
        for article in tqdm(js_array, unit='articles', ncols=120, total=len(js_array), desc = "    "):   
            if arg["lowercase"]:
                for sentence in article[arg["x_field"]]:            
                    frequency_counter+=collections.Counter([token.lower() for token in sentence])
                for sentence in article[arg["y_field"]]:
                    frequency_counter+=collections.Counter([token.lower() for token in sentence])
            else:
                for sentence in article[arg["x_field"]]:            
                    frequency_counter+=collections.Counter(sentence)
                for sentence in article[arg["y_field"]]:
                    frequency_counter+=collections.Counter(sentence)
                
        # drop last elements, otherwise everything is stuck    
        #print("Before drop: "+str(len(frequency_counter)))
        frequency_counter = collections.Counter(dict(frequency_counter.most_common(arg["max_vocab_size"])))
        json.dump(frequency_counter.most_common(), open(frequency_counter_file,"w",encoding="utf-8"))
        #for key, count in dropwhile(lambda key_count: key_count[1] >= arg["minimum_frequency"], frequency_counter.most_common()):
        #    del frequency_counter[key]
        #print("After drop: "+str(len(frequency_counter)))
    json.dump(frequency_counter.most_common(), open(frequency_counter_file,"w",encoding="utf-8"))
    print("\n\n")       
else:
    print("\t Reading word counts from file ...")
    frequency_counter = collections.Counter(dict(json.load(open(frequency_counter_file))))
    
# create word2index and index2word     
print("Creating word2index and index2word ...")
word2index = {}
index2word = {}

word2index['<pad>'] = 0
word2index['<unk>'] = 1
word2index['<s>'] = 2
word2index['</s>'] = 3
index = 4
for word in frequency_counter:
    word2index[word] = index
    index+=1
    
index2word = {word2index[word]:word for word in word2index}
json.dump(word2index, open(os.path.join(arg["output_folder"],"word2index.json"),"w",encoding="utf-8"))
json.dump(index2word, open(os.path.join(arg["output_folder"],"index2word.json"),"w",encoding="utf-8"))

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
        x = [word2index["<s>"]]
        for sentence in article[arg["x_field"]]:            
            if len(x)+len(sentence) < arg["max_sequence_len"]-1:
                x+=words2ints(sentence)
            else:
                break
        x+= [word2index["</s>"]]
            
        # process y
        y = [word2index["<s>"]]
        for index, sentence in enumerate(article[arg["y_field"]]):            
            if index>=arg["keep_max_y"]:
                break            
            y+=words2ints(sentence)            
        y+= [word2index["</s>"]]
        
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



   
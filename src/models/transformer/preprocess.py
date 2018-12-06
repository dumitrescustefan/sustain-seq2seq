"""
This script creates a vocabulary based on the input folder.

Set input parameters below:
"""
import os

arg = {}
arg["fresh_start"] = False # set to True to overwrite everything. This will take a while.

arg["input_folder"] = os.path.abspath("../../data/processed") # where the cnn and dm folders contain the processed jsons
arg["output_folder"] = os.path.abspath("../../train/transformer") # where to store the vocab dict and indexes
arg["lowercase"] = False # whether to lowercase or not
arg["max_vocab_size"] = 50000 # maximum number of words in the vocab
arg["max_sequence_len"] = 400 # max length of an instance
arg["validation_fraction"] = 0.05 # fraction to use as validation
arg["test_fraction"] = 0.05 # fraction to test on
arg["full_data_fraction"] = 0.1 # what fraction from all avaliable data to use (1.0 if you want full dataset)

arg["x_field"] = "x_tokenized_original_sentences"
arg["y_field"] = "y_tokenized_original_sentences"

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
   
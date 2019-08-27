import sys
sys.path.append("../..")

import os, json, glob, collections
from tqdm import tqdm
from itertools import dropwhile
from models.util.lookup import Lookup
import torch

""" BPE """
lookup_type = "bpe"
src_lookup_file_prefix = os.path.join("lookup","bpe","src")+"-256"
tgt_lookup_file_prefix = os.path.join("lookup","bpe","tgt")+"-256"

"""
lookup_type = "gpt2"
src_lookup_file_prefix = os.path.join("lookup","gpt2","src")
tgt_lookup_file_prefix = os.path.join("lookup","gpt2","tgt")
"""

input_src_file = os.path.join("raw","cmudict-0.7b.X.txt")
input_tgt_file = os.path.join("raw","cmudict-0.7b.y.txt")
output_folder = os.path.join("ready",lookup_type)
max_line_tokens_length = 1000
validation_fraction = 0.01
test_fraction = 0.001
full_data_fraction = 1.


# load lookups
src_lookup = lookup = Lookup(type=lookup_type)
src_lookup.load(file_prefix = src_lookup_file_prefix)
tgt_lookup = lookup = Lookup(type=lookup_type)
tgt_lookup.load(file_prefix = tgt_lookup_file_prefix)

# create output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# process files
import random

print("Creating train dev and test files ...") 
   
train_X = []
train_y = []
dev_X = []
dev_y = []
test_X = []
test_y = []   
total_len = 0

with open(input_src_file, "r", encoding="utf8") as f:
    src_lines = f.readlines()
    src_lines = [x.strip() for x in src_lines]
with open(input_tgt_file, "r", encoding="utf8") as f:
    tgt_lines = f.readlines()
    tgt_lines = [x.strip() for x in tgt_lines]

if full_data_fraction<1.:
    max_instances = int(len(src_lines)*full_data_fraction)
    print("Cutting from {} instances to {}, with data_fraction={}".format(len(src_lines),max_instances,full_data_fraction))
    src_lines = src_lines[:max_instances]
    tgt_lines = tgt_lines[:max_instances]

cnt = -1
skipped_len = 0
skipped_error = 0
for src_line, tgt_line in zip(src_lines, tgt_lines):
    cnt+=1
    if cnt%1000 == 0:
        print("{} / {} ...".format(cnt, len(src_lines)))
        
    try:    
        src_ids = src_lookup.encode(src_line, add_bos_eos_tokens=True)
        tgt_ids = tgt_lookup.encode(tgt_line, add_bos_eos_tokens=True)    
        if len(src_ids) > max_line_tokens_length or len(tgt_ids) > max_line_tokens_length:
            skipped_len += 1
            continue
    except:
        print()
        print(src_line)
        print(tgt_line)
        skipped_error += 1
        continue
    
    # select train dev or test
    train_fraction = 1.0 - validation_fraction - test_fraction
    dev_fraction = 1.0 - test_fraction
    rand = random.random()
    if rand<train_fraction:
        train_X.append(src_ids)
        train_y.append(tgt_ids)
    elif rand<dev_fraction:
        dev_X.append(src_ids)
        dev_y.append(tgt_ids)
    else:
        test_X.append(src_ids)
        test_y.append(tgt_ids)

# save train dev test
import torch
torch.save(train_X,os.path.join(output_folder,"train_X.pt"))
torch.save(train_y,os.path.join(output_folder,"train_y.pt"))
torch.save(dev_X,os.path.join(output_folder,"dev_X.pt"))
torch.save(dev_y,os.path.join(output_folder,"dev_y.pt"))
torch.save(test_X,os.path.join(output_folder,"test_X.pt"))
torch.save(test_y,os.path.join(output_folder,"test_y.pt"))
print("\n\nTrain has {} examples, dev has {}, and test has {}".format(len(train_X),len(dev_X), len(test_X)))
print("Skipped because of exceeding length {}, skipped for errors {}".format(skipped_len, skipped_error))
print("\n\nDone.")


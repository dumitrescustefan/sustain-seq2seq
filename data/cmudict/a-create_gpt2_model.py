import os, sys, json, glob, collections
sys.path.append("../..")

import torch
from models.util.lookup import Lookup
from tqdm import tqdm
from itertools import dropwhile
import sentencepiece as spm
    
input_raw_file = os.path.join("raw","cmudict-0.7b")
input_src_vocab_size = 62
input_tgt_vocab_size = input_src_vocab_size
output_lookup_folder = os.path.join("lookup","gpt2")

# create output folder
if not os.path.exists(output_lookup_folder):
    os.makedirs(output_lookup_folder)

# EXTRACT DATA FROM RAW -> TEXT

with open(input_raw_file,"r",encoding="ISO-8859-1") as f:
    content = f.readlines()
content = [x.strip() for x in content]
X_text = []
y_text = []
for line in content:
    if line.startswith(";;;"):
        continue
    parts = line.split()
    X_text.append(parts[0])
    y_text.append(parts[1:])
    
with open(input_raw_file+".X.txt","w",encoding="utf8") as f:
    for line in X_text:
        f.write(line.strip()+"\n")
        
with open(input_raw_file+".y.txt","w",encoding="utf8") as f:
    for line in y_text:                
        f.write(" ".join(line).strip()+"\n")

# CREATE LOOKUPS
src_lookup = Lookup(type="gpt2")
src_lookup.pad_token = src_lookup.eos_token
src_lookup.save_additional_tokens(file_prefix = os.path.join(output_lookup_folder,"src"))


tgt_lookup = Lookup(type="gpt2")
tgt_lookup.pad_token = src_lookup.eos_token
tgt_lookup.save_additional_tokens(file_prefix = os.path.join(output_lookup_folder,"tgt"))

print("Done.")

# check everything is ok
lookup = Lookup(type="gpt2")
lookup.load(file_prefix = os.path.join(output_lookup_folder,"tgt"))
text = " ".join(y_text[0]) # X_text[0]
print("Text: {}".format(text))
token_ids = lookup.encode(text)
print("Encode: {}".format(token_ids))
recreated_string = lookup.decode(token_ids)
print("Decode: {}".format(recreated_string))
print("Map w2i:")
tokens = lookup.tokenize(text)
for i in range(len(tokens)):    
    print("\t[{}] = [{}]".format(tokens[i], lookup.convert_tokens_to_ids(tokens[i])))

print("Map i2w:")
for i in range(len(token_ids)):
    print("\t[{}] = [{}]".format(token_ids[i], lookup.convert_ids_to_tokens(token_ids[i])))


token_ids = lookup.encode(text, add_bos_eos_tokens = True)
print("Encode with bos/eos: {}".format(token_ids))

print("\t to tokens: {}".format(lookup.convert_ids_to_tokens(token_ids))
#recreated_string = lookup.decode(token_ids)
#print("Decode with bos/eos: {}".format(recreated_string))
recreated_string = lookup.decode(token_ids, skip_bos_eos_tokens = True)
print("Decode w/o  bos/eos: {}".format(recreated_string))




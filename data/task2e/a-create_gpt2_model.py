import os, sys, json, glob, collections
sys.path.append("../..")

import torch
from models.util.lookup import Lookup
from tqdm import tqdm
from itertools import dropwhile
import sentencepiece as spm
    
output_lookup_folder = os.path.join("lookup","gpt2")

# create output folder
if not os.path.exists(output_lookup_folder):
    os.makedirs(output_lookup_folder)

# CREATE LOOKUPS
src_lookup = Lookup(type="gpt2")
src_lookup.save_special_tokens(file_prefix = os.path.join(output_lookup_folder,"src"))

tgt_lookup = Lookup(type="gpt2")
tgt_lookup.save_special_tokens(file_prefix = os.path.join(output_lookup_folder,"tgt"))

print("Done.")

# check everything is ok
lookup = Lookup(type="gpt2")
lookup.load(file_prefix = os.path.join(output_lookup_folder,"tgt"))
text = "This is a test."
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
recreated_string = lookup.decode(token_ids)
print("Decode with bos/eos: {}".format(recreated_string))
recreated_string = lookup.decode(token_ids, skip_bos_eos_tokens = True)
print("Decode w/o  bos/eos: {}".format(recreated_string))




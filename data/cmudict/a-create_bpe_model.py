import os, sys, json, glob, collections
sys.path.append("../..")

import torch
from models.util.lookup import Lookup
from tqdm import tqdm
from itertools import dropwhile
import sentencepiece as spm
    
input_raw_file = os.path.join("raw","cmudict-0.7b")
input_src_vocab_size = 256
input_tgt_vocab_size = input_src_vocab_size
output_lookup_folder = os.path.join("lookup","bpe")

# create output folder
if not os.path.exists(output_lookup_folder):
    os.makedirs(output_lookup_folder)

# EXTRACT DATA FROM RAW -> TEXT

with open(input_raw_file) as f:
    content = f.readlines()
content = [x.strip() for x in content]
X_text = []
y_text = []
for line in content:
    if line.startswith(";;;"):
        continue
    parts = line.split()
    X_text.append(parts[0])
    X_text.append(" ".join(parts[1:])) # WE APPEND y as well to cover all vocab chars
    y_text.append(parts[1:])

with open(input_raw_file+".X.txt","w",encoding="utf8") as f:
    for line in X_text:
        f.write(line.strip()+"\n")
        
with open(input_raw_file+".y.txt","w",encoding="utf8") as f:
    for line in y_text:        
        f.write(" ".join(line).strip()+"\n")
    
with open(input_raw_file+".Xy.txt","w",encoding="utf8") as f:
    for line in X_text:
        f.write(line.strip()+"\n")
        
# TRAIN SENTENCEPIECE MODELS
spm.SentencePieceTrainer.Train('--input='+input_raw_file+'.Xy.txt --model_prefix='+os.path.join(output_lookup_folder,"src-"+str(input_src_vocab_size))+' --character_coverage=1.0 --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size='+str(input_src_vocab_size))
spm.SentencePieceTrainer.Train('--input='+input_raw_file+'.Xy.txt --model_prefix='+os.path.join(output_lookup_folder,"tgt-"+str(input_tgt_vocab_size))+' --character_coverage=1.0 --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size='+str(input_tgt_vocab_size))
#--pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS>  

# CREATE LOOKUPS
src_lookup = Lookup(type="bpe")
src_lookup.bos_token = '<BOS>'
src_lookup.eos_token = '<EOS>'
src_lookup.unk_token = '<UNK>'
src_lookup.sep_token = '<SEP>'
src_lookup.pad_token = '<PAD>'
src_lookup.cls_token = '<CLS>'
src_lookup.mask_token = '<MSK>'
src_lookup.save_additional_tokens(file_prefix = os.path.join(output_lookup_folder,"src-"+str(input_src_vocab_size)))
#src_lookup.load(os.path.join(output_lookup_folder,"src-"+str(input_src_vocab_size)))

tgt_lookup = Lookup(type="bpe")
tgt_lookup.bos_token = '<BOS>'
tgt_lookup.eos_token = '<EOS>'
tgt_lookup.unk_token = '<UNK>'
tgt_lookup.sep_token = '<SEP>'
tgt_lookup.pad_token = '<PAD>'
tgt_lookup.cls_token = '<CLS>'
tgt_lookup.mask_token = '<MSK>'
tgt_lookup.save_additional_tokens(file_prefix = os.path.join(output_lookup_folder,"tgt-"+str(input_tgt_vocab_size)))
print("Done.")

# check everything is ok
lookup = Lookup(type="bpe")
lookup.load(file_prefix = os.path.join(output_lookup_folder,"tgt-"+str(input_tgt_vocab_size)))

text = X_text[0]#" ".join(y_text[0]) # X_text[0]
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
recreated_string = lookup.decode(token_ids, skip_bos_eos_tokens = True)
print("Decode w/o  bos/eos: {}".format(recreated_string))



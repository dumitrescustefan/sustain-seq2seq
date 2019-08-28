import os, sys, json, glob, collections
sys.path.append("../..")

import torch
from models.util.lookup import Lookup
from tqdm import tqdm
from itertools import dropwhile
import sentencepiece as spm
    
input_src_file = os.path.join("raw","JRC-Acquis.en-fr.fr")
input_tgt_file = os.path.join("raw","JRC-Acquis.en-fr.en")
input_src_vocab_size = 4096
input_tgt_vocab_size = input_src_vocab_size
output_lookup_folder = os.path.join("lookup","bpe")

# create output folder
if not os.path.exists(output_lookup_folder):
    os.makedirs(output_lookup_folder)

# TRAIN SENTENCEPIECE MODELS
spm.SentencePieceTrainer.Train('--input='+input_src_file+' --model_prefix='+os.path.join(output_lookup_folder,"src-"+str(input_src_vocab_size))+' --character_coverage=1.0 --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --max_sentence_length=8000 --vocab_size='+str(input_src_vocab_size))
spm.SentencePieceTrainer.Train('--input='+input_tgt_file+' --model_prefix='+os.path.join(output_lookup_folder,"tgt-"+str(input_tgt_vocab_size))+' --character_coverage=1.0 --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --max_sentence_length=8000 --vocab_size='+str(input_tgt_vocab_size))

# CREATE LOOKUPS
src_lookup = Lookup(type="bpe")
src_lookup.save_special_tokens(file_prefix = os.path.join(output_lookup_folder,"src-"+str(input_src_vocab_size)))
#src_lookup.load(os.path.join(output_lookup_folder,"src-"+str(input_src_vocab_size)))

tgt_lookup = Lookup(type="bpe")
tgt_lookup.save_special_tokens(file_prefix = os.path.join(output_lookup_folder,"tgt-"+str(input_tgt_vocab_size)))

print("Done.")

# check everything is ok
lookup = Lookup(type="bpe")
lookup.load(file_prefix = os.path.join(output_lookup_folder,"tgt-"+str(input_tgt_vocab_size))) # "This is a test."
text = "Износителят на продуктите, обхванати от този документ (митническо разрешение No … (1)) декларира, че освен кьдето е отбелязано друго, тези продукти са с … (2) преференциален произход . "

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



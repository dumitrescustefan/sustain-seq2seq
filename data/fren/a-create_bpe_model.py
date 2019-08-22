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
input_tgt_vocab_size = 4096
output_lookup_folder = os.path.join("lookup","bpe")

# create output folder
if not os.path.exists(output_lookup_folder):
    os.makedirs(output_lookup_folder)

# TRAIN SENTENCEPIECE MODELS
spm.SentencePieceTrainer.Train('--input='+input_src_file+' --model_prefix='+os.path.join(output_lookup_folder,"src-"+str(input_src_vocab_size))+' --character_coverage=1.0 --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size='+str(input_src_vocab_size))
spm.SentencePieceTrainer.Train('--input='+input_tgt_file+' --model_prefix='+os.path.join(output_lookup_folder,"tgt-"+str(input_tgt_vocab_size))+' --character_coverage=1.0 --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size='+str(input_tgt_vocab_size))

# CREATE LOOKUPS
src_lookup = Lookup(type="bpe")
src_lookup.load(os.path.join(output_lookup_folder,"src-"+str(input_src_vocab_size)))
src.save(os.path.join(output_lookup_folder,"src-"+str(input_src_vocab_size)))

tgt_lookup = Lookup(type="bpe")
tgt_lookup.load(os.path.join(output_lookup_folder,"tgt-"+str(input_src_vocab_size)))
tgt.save(os.path.join(output_lookup_folder,"tgt-"+str(input_src_vocab_size)))

print("Done.")
import os, sys, json, glob, collections
sys.path.append("../..")

import torch
from models.util.lookup import Lookup
from tqdm import tqdm
from itertools import dropwhile
import sentencepiece as spm

input_folder = "raw"    
input_json_file = os.path.join(input_folder,"data.json")
input_src_vocab_size = 1024
input_tgt_vocab_size = input_src_vocab_size
output_lookup_folder = os.path.join("lookup","bpe")

# create output folder
if not os.path.exists(output_lookup_folder):
    os.makedirs(output_lookup_folder)

# create text files
data = json.load(open(input_json_file,"r",encoding="utf8"))
with open(os.path.join(input_folder,"all.txt"), "w", encoding="utf8") as all_f:
    for MEI in data:
        with open(os.path.join(input_folder,MEI.replace(" ","_")+".txt"), "w", encoding="utf8") as f:
            for cpy in data[MEI]:
                if data[MEI][cpy]["modified"] == "yes":
                    for sentence in data[MEI][cpy]["sentences"]:
                        f.write(sentence+"\n")
                        all_f.write(sentence+"\n")
                    f.write(data[MEI][cpy]["output"]+"\n")
                    all_f.write(data[MEI][cpy]["output"]+"\n")
   
# TRAIN SENTENCEPIECE MODELS & CREATE LOOKUPS
for MEI in data:
    MEI = MEI.replace(" ","_")
    print("Prep BPE train for : "+MEI)
    try:    
        spm.SentencePieceTrainer.Train('--input='+os.path.join(input_folder, MEI+".txt")+' --model_prefix='+os.path.join(output_lookup_folder,"src-"+MEI+"-"+str(input_src_vocab_size))+' --character_coverage=1.0 --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --max_sentence_length=8000 --vocab_size='+str(input_src_vocab_size))
        print("Done.")
        src_lookup = Lookup(type="bpe")
        src_lookup.save_special_tokens(file_prefix = os.path.join(output_lookup_folder,"src-"+MEI+"-"+str(input_src_vocab_size)))
    except:
        print("ERROR, skipping "+MEI)
        
spm.SentencePieceTrainer.Train('--input='+os.path.join(input_folder,"all.txt")+' --model_prefix='+os.path.join(output_lookup_folder,"src-"+str(input_src_vocab_size))+' --character_coverage=1.0 --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --max_sentence_length=8000 --vocab_size='+str(input_src_vocab_size))
src_lookup = Lookup(type="bpe")
src_lookup.save_special_tokens(file_prefix = os.path.join(output_lookup_folder,"src-"+str(input_src_vocab_size)))
    
print("Done.")

# check everything is ok
lookup = Lookup(type="bpe")
lookup.load(file_prefix = os.path.join(output_lookup_folder,"src-"+str(input_src_vocab_size))) # "This is a test."
text = "This company."

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



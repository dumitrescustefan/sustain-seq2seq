import os, sys, json, glob, collections
sys.path.append("../..")

import torch
from models.util.lookup import Lookup
from tqdm import tqdm
from itertools import dropwhile
import sentencepiece as spm
    
input_src_file = os.path.join("raw","JRC-Acquis.en-fr.fr")
input_tgt_file = os.path.join("raw","JRC-Acquis.en-fr.en")
input_src_vocab_size = 2048
input_tgt_vocab_size = 2048
output_lookup_folder = os.path.join("lookup","bpe")
validation_fraction = 0.005
test_fraction = 0.0125
full_data_fraction = 1.

# create output folder
if not os.path.exists(output_lookup_folder):
    os.makedirs(output_lookup_folder)

# CREATE BPE MODELS
spm.SentencePieceTrainer.Train('--input='+input_src_file+' --model_prefix='+os.path.join(output_lookup_folder,"src-"+str(input_src_vocab_size))+' --character_coverage=1.0 --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size='+input_src_vocab_size)
spm.SentencePieceTrainer.Train('--input='+input_tgt_file+' --model_prefix='+os.path.join(output_lookup_folder,"tgt-"+str(input_tgt_vocab_size))+' --character_coverage=1.0 --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size='+input_tgt_vocab_size)

# CREATE LOOKUPS
src_lookup = Lookup(type="bpe")
src_lookup.load(os.path.join(output_lookup_folder,"src-"+str(input_src_vocab_size)))

# create word2index and index2word     
print("Creating word2index and index2word ...")
fr_word2index = {}
fr_index2word = {}
en_word2index = {}
en_index2word = {}

index = -1

with open(bpe_model_vocab_fr"],"r",encoding="utf8") as f:    
    for line in f:
        index+=1
        word = line.split("\t")[0]
        fr_word2index[word] = index
        fr_index2word[str(index)] = word

index = -1        
with open(bpe_model_vocab_en"],"r",encoding="utf8") as f:    
    for line in f:
        index+=1
        word = line.split("\t")[0]
        en_word2index[word] = index
        en_index2word[str(index)] = word
        
        
# just to be safe, overwrite special markers
fr_word2index['<PAD>'] = 0
fr_word2index['<UNK>'] = 1
fr_word2index['<BOS>'] = 2
fr_word2index['<EOS>'] = 3
fr_index2word['0']='<PAD>'
fr_index2word['1']='<UNK>'
fr_index2word['2']='<BOS>'
fr_index2word['3']='<EOS>'
en_word2index['<PAD>'] = 0
en_word2index['<UNK>'] = 1
en_word2index['<BOS>'] = 2
en_word2index['<EOS>'] = 3
en_index2word['0']='<PAD>'
en_index2word['1']='<UNK>'
en_index2word['2']='<BOS>'
en_index2word['3']='<EOS>'

json.dump(fr_word2index, open(os.path.join(output_folder"],"fr_word2index.json"),"w",encoding="utf-8"), sort_keys=True)
json.dump(fr_index2word, open(os.path.join(output_folder"],"fr_index2word.json"),"w",encoding="utf-8"), sort_keys=True)

json.dump(en_word2index, open(os.path.join(output_folder"],"en_word2index.json"),"w",encoding="utf-8"), sort_keys=True)
json.dump(en_index2word, open(os.path.join(output_folder"],"en_index2word.json"),"w",encoding="utf-8"), sort_keys=True)


# process files
import random
import sentencepiece as spm
sp_fr = spm.SentencePieceProcessor()
sp_fr.Load(bpe_model_model_fr)
sp_en = spm.SentencePieceProcessor()
sp_en.Load(bpe_model_model_en)

print("Creating train dev and test files ...") 
   
train_X = []
train_y = []
dev_X = []
dev_y = []
test_X = []
test_y = []   
total_len = 0

with open(input_folder"]+"/JRC-Acquis.en-fr.fr", "r", encoding="utf8") as f:
    fr_lines = f.readlines()
    fr_lines = [x.strip() for x in fr_lines]
with open(input_folder"]+"/JRC-Acquis.en-fr.en", "r", encoding="utf8") as f:
    en_lines = f.readlines()
    en_lines = [x.strip() for x in en_lines]


if full_data_fraction"]<1.:
    max_instances = int(len(fr_lines)*full_data_fraction"])
    print("Cutting from {} instances to {}, with data_fraction={}".format(len(fr_lines),max_instances,full_data_fraction"]))
    fr_lines = fr_lines[:max_instances]
    en_lines = en_lines[:max_instances]
    
for r, e in zip(fr_lines, en_lines):
    ri = sp_fr.EncodeAsIds(r)
    re = sp_en.EncodeAsIds(e)
    
    if reverse_x"]:
        re = re[::-1]

    x = [fr_word2index["<BOS>"]] + ri + [fr_word2index["<EOS>"]]
    y = [en_word2index["<BOS>"]] + re + [en_word2index["<EOS>"]]

    total_len += len(re)
    # select train dev or test
    train_fraction = 1.0 - validation_fraction"] - test_fraction"]
    dev_fraction = 1.0 - test_fraction"]
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
json.dump(arg, open(os.path.join(output_folder"],"preprocess_settings.json"),"w",encoding="utf-8"), indent=4, sort_keys=True)

# save train dev test
import torch
torch.save(train_X,os.path.join(output_folder"],"train_X.pt"))
torch.save(train_y,os.path.join(output_folder"],"train_y.pt"))
torch.save(dev_X,os.path.join(output_folder"],"dev_X.pt"))
torch.save(dev_y,os.path.join(output_folder"],"dev_y.pt"))
torch.save(test_X,os.path.join(output_folder"],"test_X.pt"))
torch.save(test_y,os.path.join(output_folder"],"test_y.pt"))
print("\n\nTrain has {} examples, dev has {}, and test has {}".format(len(train_X),len(dev_X), len(test_X)))
print("\n\nDone.")


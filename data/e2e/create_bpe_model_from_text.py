"""
usage: from e2e_lstm_att.create_bpe_model_from_text import e2e_create_bpe_model
e2e_create_bpe_model(output_model_folder_path):
will create a BPE model placed in the output_model_folder_path, with a json w2i and i2w files there
"""

# add package root
import os, sys, json, string
sys.path.insert(0, '../..')

import sentencepiece as spm

from data.e2e.data import Slot, Slots
from data.e2e.raw_to_slots import e2e_read

# read files
def e2e_create_bpe_model(output_model_folder_path, vocab_size):
    print("Creating "+output_model_folder_path+" ...")
    if not os.path.exists(output_model_folder_path):
        os.makedirs(output_model_folder_path)

    print("Reading all data ...")
    slots, (train_X, train_y), (dev_X, dev_y), (test_X, test_y)= e2e_read()

    print("Writing all text ...")
    with open(output_model_folder_path+"/temp.txt", "w", encoding="utf8") as f:
        for line in train_y:
            f.write(line.strip()+"\n")
        f.write(string.punctuation+"\n")
    print("Training BPE model with vocab_size="+str(vocab_size)+" ...")
    
    # we need to input control_symbols for them to have an ordered ID in the vocabulary!!
    spm.SentencePieceTrainer.Train('--input='+output_model_folder_path+'/temp.txt --model_prefix='+output_model_folder_path+'/e2e.bpe --unk_surface=[UNK] --character_coverage=1.0 --pad_id=0 --pad_piece=[PAD] --unk_id=1 --unk_piece=[UNK] --bos_id=2 --bos_piece=[BOS] --eos_id=3 --eos_piece=[EOS] --control_symbols=[CLS],[SEP],[MASK] --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size='+str(vocab_size))
    
    print("Cleaning up temp.txt ...")
    os.remove(output_model_folder_path+'/temp.txt')
 
    print("Testing ...")
    sp = spm.SentencePieceProcessor()
    sp.load(output_model_folder_path+"/e2e.bpe.model")
    ids = sp.encode_as_pieces('[CLS][BOS]A test.[EOS][SEP]Near Burger King is a coffee shop called Blue Spice which has an average customer rating.[PAD][PAD]')
    print(ids)
    
    print("Creating word2index and index2word ...")
    word2index = {}
    index2word = {}

    index = -1

    with open(output_model_folder_path+"/e2e.bpe.vocab","r",encoding="utf8") as f:    
        for line in f:
            index+=1
            word = line.split("\t")[0]
            word2index[word] = index
            index2word[str(index)] = word
    
    for token in (['[PAD]','[UNK]','[BOS]','[EOS]','[CLS]','[SEP]','[MASK]']):
        print("\tSpecial token {} = {}".format(token, sp.piece_to_id(token)))
    
    
    json.dump(word2index, open(output_model_folder_path+"/w2i.json","w",encoding="utf-8"), indent = 2, sort_keys=False)
    json.dump(index2word, open(output_model_folder_path+"/i2w.json","w",encoding="utf-8"), indent = 2, sort_keys=False)
    
    
    print("Done.")
    
if __name__=="__main__":
    e2e_create_bpe_model(output_model_folder_path="bpe", vocab_size = 1024)
    
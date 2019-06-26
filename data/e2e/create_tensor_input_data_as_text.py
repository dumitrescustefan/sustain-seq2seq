"""
usage: from e2e_lstm_att.create_bpe_model_from_text import e2e_create_tensor_input_data
e2e_create_tensor_input_data(output_data_folder_path="../data/e2e")
will create train dev and test files
"""

# add package root
import os, sys, json
sys.path.insert(0, '..')

import sentencepiece as spm

from common.dataset import Slot, Slots
from e2e_lstm_att.e2e_raw_to_slots import e2e_read
import torch

def encode_to_int (X, y, sp, w2i):    
    processed_X = []
    processed_y = []
    
    # proces X    
    print("\tProcess X ...")
    
    for i, slots in enumerate(X):
        text = []
        for (slot_name, slot_value) in slots:
            text.append(slot_name+"="+slot_value)
        print_me = True if sum([1 if elem == 1 else 0 for elem in sp.EncodeAsIds("|".join(text))])>0 else False
        if i==1 or print_me:
            print("Example: "+"-"*50)
            print(text)
            print(sp.encode_as_pieces("|".join(text)))
            print(sp.EncodeAsIds("|".join(text)))
            print_me = False
        text = [w2i["[BOS]"]]+sp.EncodeAsIds("|".join(text))+[w2i["[EOS]"]]
        if i==1:
            print(text)            
            
        processed_X.append(text)
   
    # process y 
    print("\tProcess y ...")
    for i, text in enumerate(y):
        if i==1:
            print(text)
        processed_y.append([w2i["[BOS]"]]+sp.EncodeAsIds(text.strip())+[w2i["[EOS]"]])
        if i==1:
            print(processed_y[i])
            print("-"*80)
    return processed_X, processed_y
    
def e2e_create_tensor_input_data(output_data_folder_path="../data/e2e"):
    print("Reading all data ...")
    slots, (train_X, train_y), (dev_X, dev_y), (test_X, test_y) = e2e_read()

    print("Loading BPE model ...")
    sp = spm.SentencePieceProcessor()
    sp.load(output_data_folder_path+"/e2e.bpe.model")
        
    w2i = json.load(open(output_data_folder_path+"/w2i.json","r",encoding="utf-8"))
    
    print("Train converting ...")
    train_X, train_y = encode_to_int(train_X, train_y, sp, w2i) #train_X and train_y are [ints]        
    torch.save(train_X,output_data_folder_path+"/train_X.pt")
    torch.save(train_y,output_data_folder_path+"/train_y.pt")
    
    print("Dev converting ...")
    dev_X, dev_y = encode_to_int(dev_X, dev_y, sp, w2i) #train_X and train_y are [ints]        
    torch.save(dev_X,output_data_folder_path+"/dev_X.pt")
    torch.save(dev_y,output_data_folder_path+"/dev_y.pt")
    
    print("Train converting ...")
    test_X, test_y = encode_to_int(test_X, test_y, sp, w2i) #train_X and train_y are [ints]        
    torch.save(test_X,output_data_folder_path+"/test_X.pt")
    torch.save(test_y,output_data_folder_path+"/test_y.pt")
    
    print("Done.")


    
if __name__=="__main__":
    e2e_create_tensor_input_data(output_data_folder_path="../data/e2e")
    
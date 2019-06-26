"""
usage: from create_bpe_model_from_text import create_tensor_input_data
create_tensor_input_data(output_data_folder_path="../data/e2e")
will create train dev and test files
"""

# add package root
import os, sys, json
sys.path.insert(0, '../..')

import sentencepiece as spm

from data.e2e.data import Slot, Slots
from data.e2e.raw_to_slots import e2e_read
import torch

def encode_to_int (X, y, sp, w2i, slots):    
    processed_X = []
    processed_y = []
    
    # proces X    
    print("\tProcess X ...")
    
    for i, slot_tuples in enumerate(X):
        ints = []        
        for slot in slots.slots: # ensure fixed order of slots    
            current_slot_name = slot.name            
            found = False
            # search for current_slot_name in tuples
            for (slot_name, slot_value) in slot_tuples:
                if slot_name == current_slot_name:
                    found = True                    
                    index = slot.values.index(slot_value)
                    break
            if not found:
                index = 0 # default , not-present
            ints.append(index)                
        processed_X.append(ints)
        
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
    
def create_tensor_input_data(output_data_folder_path="."):
    print("Reading all data ...")
    slots, (train_X, train_y), (dev_X, dev_y), (test_X, test_y) = e2e_read(reject_duplicates = True)

    print("Writing slots object ...")
    torch.save(slots,output_data_folder_path+"/slots.pt")    
    
    print("Loading BPE model ...")
    sp = spm.SentencePieceProcessor()
    sp.load("bpe/e2e.bpe.model")
        
    w2i = json.load(open("bpe/w2i.json","r",encoding="utf-8"))
    
    print("Train converting ...")
    train_X, train_y = encode_to_int(train_X, train_y, sp, w2i, slots) #train_X and train_y are [ints]        
    torch.save(train_X,output_data_folder_path+"/train_X.pt")
    torch.save(train_y,output_data_folder_path+"/train_y.pt")
    
    print("Dev converting ...")
    dev_X, dev_y = encode_to_int(dev_X, dev_y, sp, w2i, slots) #train_X and train_y are [ints]        
    torch.save(dev_X,output_data_folder_path+"/dev_X.pt")
    torch.save(dev_y,output_data_folder_path+"/dev_y.pt")
    
    print("Train converting ...")
    test_X, test_y = encode_to_int(test_X, test_y, sp, w2i, slots) #train_X and train_y are [ints]        
    torch.save(test_X,output_data_folder_path+"/test_X.pt")
    torch.save(test_y,output_data_folder_path+"/test_y.pt")
    
    print("Done.")

    #print(train_X[0])
    #print(train_X[1])
    #print(train_X[-2])
    
if __name__=="__main__":
    create_tensor_input_data(output_data_folder_path="vector")
    
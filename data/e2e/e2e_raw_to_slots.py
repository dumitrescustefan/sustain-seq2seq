"""
This package reads e2e format files and returns a Slots object with a train dev and test array of X and Ys
ex: "name[Alimentum], area[city centre], familyFriendly[no]","There is a place in the city centre, Alimentum, that is not family-friendly."

usage: from v2.e2e_raw_to_slots import e2e_read
slots, (train_X, train_y), (dev_X, dev_y), (test_X, test_y) = e2e_read()
train_X is a list of [(slot_name, slot_value),...]
train_y is a text ""
"""

# add package root
import os, sys
sys.path.insert(0, '..')

from common.dataset import Slot, Slots

def read_mr_file (file):
    with open(file, "r", encoding="utf8") as f:
        X = []
        y = []
        for line_index, line in enumerate(f):
            if len(line)<20:
                continue
            #print("|"+line+"|")
            line = line.strip()
            try:
                middle = line.strip().index('",')
                slots = line[:middle]
                if slots.startswith('"'):
                    slots = slots[1:]
                s = []
                slot_array = slots.split(",")
                for slot in slot_array:
                    text = slot.strip()
                    start = text.index("[")
                    slot_name = text[:start]
                    slot_value = text[start+1:-1]
                    s.append((slot_name, slot_value))
                X.append(s)
                
                text = line[middle+2:]
                if text.startswith("\""):
                    text=text[1:]
                if text.endswith("\""):
                    text=text[:-1]
                y.append(text)
            except:
                print("Exception: |"+line+"| in "+file+", line "+str(line_index)) 
            #print(X)
            #print(y)
            #break        
        return X, y
        
# read files
def e2e_read():
    slots = Slots(mei = "e2e")
    train_X, train_y = read_mr_file("../data/e2e/raw/trainset.csv")
    dev_X, dev_y = read_mr_file("../data/e2e/raw/devset.csv")
    test_X, test_y = read_mr_file("../data/e2e/raw/testset_w_refs.csv")

    for x in train_X + dev_X + test_X:
        for (slot_name, slot_value) in x:                
            slots.add(slot_name, slot_value)
    
    #print(slots)
    return slots, (train_X, train_y), (dev_X, dev_y), (test_X, test_y)
    
if __name__=="__main__":
    e2e_read()
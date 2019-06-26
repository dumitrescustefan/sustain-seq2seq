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
sys.path.insert(0, '../..')

from data.e2e.data import Slot, Slots

def read_mr_file_old (file, reject_duplicates = False):
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

def read_mr_file (file, reject_duplicates = False):
    with open(file, "r", encoding="utf8") as f:
        X = []
        y = []
        data = {}
        original_tuples = {}
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
                if str(s) not in data:
                    data[str(s)] = []
                    original_tuples[str(s)] = s
                
                text = line[middle+2:]
                if text.startswith("\""):
                    text=text[1:]
                if text.endswith("\""):
                    text=text[:-1]
                data[str(s)].append(text)                
            except:
                print("Exception: |"+line+"| in "+file+", line "+str(line_index)) 
        
        if reject_duplicates:            
            for s in data:
                texts = data[s]
                longest_index = 0
                longest_string = len(texts[0])
                for i in range (len(texts)):                    
                    if len(texts[i]) > longest_string:
                        longest_index = i
                X.append(original_tuples[s])
                y.append(texts[longest_index])
        else:
            for s in data:
                texts = data[s]
                for i in range(len(texts)):
                    X.append(original_tuples[s])
                    y.append(texts[i])
        return X, y


        
# read files
def e2e_read(reject_duplicates=False):
    slots = Slots()
    train_X, train_y = read_mr_file("raw/trainset.csv", reject_duplicates)
    dev_X, dev_y = read_mr_file("raw/devset.csv", reject_duplicates)
    test_X, test_y = read_mr_file("raw/testset_w_refs.csv", reject_duplicates)
    
    for x in train_X + dev_X + test_X:                
        for (slot_name, slot_value) in x:                            
            slots.add_slot_value_pair(slot_name, slot_value)
    
    return slots, (train_X, train_y), (dev_X, dev_y), (test_X, test_y) 
    
if __name__=="__main__":    
    slots, (train_X, train_y), (dev_X, dev_y), (test_X, test_y) = e2e_read(reject_duplicates = False)    
    print(slots)
    print("Train/dev/test lenghts: {}/{}/{}".format(len(train_X), len(dev_X), len(test_X)))
    print(train_X[0])
    print(train_y[0])
    
    
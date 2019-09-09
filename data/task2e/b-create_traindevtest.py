import sys
sys.path.append("../..")

import os, json, glob, collections
from tqdm import tqdm
from itertools import dropwhile
from models.util.lookup import Lookup
import torch

if len(sys.argv)!=2:
    print("param: bpe or gpt2")
    sys.exit(0)

input_json_file = os.path.join("raw","data.json")
max_line_tokens_length = 1000
validation_fraction = 0.075
test_fraction = 0.0

# find out all MEIs available
import glob, ntpath
files = glob.glob(os.path.join("raw","*.txt"))
MEIs = []
for file in files:
    MEIs.append(ntpath.basename(file).replace(".txt","").replace("_"," "))

print("Available MEIs:")
print(MEIs)
print()

for MEI in MEIs:
    if MEI == "all":
        continue
    print("\nWorking on "+MEI)
    
    if sys.argv[1] == "bpe":
        lookup_type = "bpe"        
        fname = "-"+MEI.replace(" ","_")
        src_lookup_file_prefix = os.path.join("lookup","bpe","src"+fname+"-1024")
        tgt_lookup_file_prefix = os.path.join("lookup","bpe","src"+fname+"-1024")        

    if sys.argv[1] == "gpt2":
        lookup_type = "gpt2"
        src_lookup_file_prefix = os.path.join("lookup","gpt2","src")
        tgt_lookup_file_prefix = os.path.join("lookup","gpt2","tgt")

    # load lookups
    try:
        src_lookup = Lookup(type=lookup_type)
        src_lookup.load(file_prefix = src_lookup_file_prefix)
        tgt_lookup = Lookup(type=lookup_type)
        tgt_lookup.load(file_prefix = tgt_lookup_file_prefix)
    except:
        print("ERROR with "+src_lookup_file_prefix)
        continue

    data = json.load(open(input_json_file,"r",encoding="utf8"))
    
    output_folder = os.path.join("ready",lookup_type)    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # process files
    import random

    print("\tInfering slots from data file...")
    slots_dict = {}
    slots_encoding = {}
    for mei in data:
        print(mei)
        if mei not in slots_dict:
            slots_dict[mei] = {}
        for cpy_id in data[mei]:
            cpy = data[mei][cpy_id]
            slots = cpy["slots"]
            slot_list = []
            for slot in slots: 
                if "FAILED" in slot:
                    continue
                    #print(slot)
                    #sys.exit(0)
                    #slot = slot.replace("FAILED ","").strip().replace("[","").replace("]","")
                    #slot = "("+slot+", 'N/A-FAILED')"
                if slot[0] != "[":
                    slot = "["+slot+"]"
                #print(slot)
                #print(slot)
                to_list = eval(slot)#json.loads(slot)
                slot_list+=to_list
            # append to data object
            data[mei][cpy_id]["slots_list"] = slot_list
            #print()
            #print(slot_list)
            
            for slot_tuple in slot_list:
                #print(slot_tuple)
                slot_name, slot_value = slot_tuple[0], slot_tuple[1]
                if slot_name not in slots_dict[mei]:
                    slots_dict[mei][slot_name] = []
                if slot_value not in slots_dict[mei][slot_name]:
                    slots_dict[mei][slot_name].append(slot_value)
                slots_dict[mei][slot_name] = sorted(slots_dict[mei][slot_name])
                
        print(slots_dict[mei])
        print()
        
        names_only = []
        for slot_name in slots_dict[mei]:
            names_only.append(slot_name)
        slots_encoding[mei] = sorted(names_only)
        
    #sys.exit(0)

    print("\tCreating train dev and test files ...") 
       
    train_X = []
    train_y = []
    train_slots = []
    dev_X = []
    dev_y = []
    dev_slots = []
    test_X = []
    test_y = []   
    test_slots = []
    total_len = 0

    cpy_list = []
    if MEI == "all":        
        continue
    else:        
        for cpy_id in data[MEI]:
            cpy_list.append(data[MEI][cpy_id])
            cpy_list[-1]["cpy_id"] = cpy_id
            
    print("\tCpy pool = {}".format(len(cpy_list)))
    
    cnt = -1
    skipped_len = 0
    skipped_error = 0
    skipped_modified = 0
    for j in range(len(cpy_list)):        
        if cpy_list[j]["modified"]!="yes":
            skipped_modified += 1
            continue        
    
        sentences = cpy_list[j]["sentences"]
        order = cpy_list[j]["order"]
        output = cpy_list[j]["output"]
        cpy_id = cpy_list[j]["cpy_id"]
        slots_list = cpy_list[j]["slots_list"]
        
        try:
            new_order = []
            if order[0]=="": # convert to int or generate sequence if order == ""
                for i in range(len(sentences)):
                    new_order.append(int(i))
            else:
                #print(cpy_id)
                #print(order)            
                for i in range(len(sentences)):                                
                    if order[i] != "":
                        new_order.append(int(float(order[i]))-1)
                    else:
                        new_order.append(-1)
                #print(new_order)
                #print()
            
            src_line = ""
            for i in range(len(new_order)):                 
                if i in new_order:
                    index = new_order.index(i)
                    src_line += " "+sentences[index]
            src_line = src_line.strip()
            
            """print()
            print()
            print(order)
            print(new_order)
            
            if sentences[-1] != sentences[index]:
                print(sentences)
                print("-----------")
                print(output)
                input("  ")
            """
            
            tgt_line = output.strip()
            
            # gen slot list encoding             
            slots = []
            for i in range(len(slots_encoding[MEI])):
                slot_name = slots_encoding[MEI][i]
                #print(slots_dict[MEI][slot_name])
                # get value for this slot
                
                # first, search in list
                found = False
                #print(slot_name)
                for j in range(len(slots_list)):                    
                    my_name = slots_list[j][0] # name is first in tuple
                    my_value = slots_list[j][1]
                    #print("\t "+my_name+" = "+my_value)
                    
                    if my_name == slot_name:
                        found = True
                        try:
                            slot_value = 1 + slots_dict[MEI][slot_name].index(my_value)
                        except:
                            slot_value = 0                             
                        break
                if not found:
                    slot_value = 0
                slots.append(slot_value)
            #print(slots)
            #sys.exit(0)
        except KeyboardInterrupt:
            sys.exit(0)
        #except:
        #    skipped_error += 1
        #    continue
            
        cnt+=1
        if cnt%100 == 0:
            print("{} / {} ...".format(cnt, len(cpy_list)))
            
        try:    
            if sys.argv[1] == "bpe":
                src_ids = src_lookup.encode(src_line, add_bos_eos_tokens=True)
            if sys.argv[1] == "gpt2":
                src_ids = src_lookup.encode(src_line, add_bos_eos_tokens=False)
            
            tgt_ids = tgt_lookup.encode(tgt_line, add_bos_eos_tokens=True)    
            
            if cnt%100 == 0:
                print("\n+++++++SRC:")
                print(src_line)
                print(src_ids)
                print(src_lookup.decode(src_ids))
                print(src_lookup.decode(src_ids, skip_bos_eos_tokens=True))
                print("+++++++TGT")
                print(tgt_line)
                print(tgt_ids)
                print(tgt_lookup.decode(tgt_ids))
                print(tgt_lookup.decode(tgt_ids, skip_bos_eos_tokens=True))
                print("+++++++\n")
                
            if len(src_ids) > max_line_tokens_length or len(tgt_ids) > max_line_tokens_length:
                skipped_len += 1
                continue
        except:
            print()
            print(src_line)
            print(tgt_line)
            skipped_error += 1
            continue
        
        # select train dev or test
        train_fraction = 1.0 - validation_fraction - test_fraction
        dev_fraction = 1.0 - test_fraction
        rand = random.random()
        if rand<train_fraction:
            train_X.append(src_ids)
            train_y.append(tgt_ids)
            train_slots.append(slots)
        elif rand<dev_fraction:
            dev_X.append(src_ids)
            dev_y.append(tgt_ids)
            dev_slots.append(slots)
        else:
            test_X.append(src_ids)
            test_y.append(tgt_ids)
            test_slots.append(slots)

    # save train dev test
    import torch    
    torch.save(train_X,os.path.join(output_folder, MEI.replace(" ","_")+"_train_X.pt"))
    torch.save(train_y,os.path.join(output_folder, MEI.replace(" ","_")+"_train_y.pt"))    
    torch.save(train_slots,os.path.join(output_folder, MEI.replace(" ","_")+"_train_slots.pt"))    
    if len(dev_X) > 0:
        torch.save(dev_X,os.path.join(output_folder, MEI.replace(" ","_")+"_dev_X.pt"))
        torch.save(dev_y,os.path.join(output_folder, MEI.replace(" ","_")+"_dev_y.pt"))    
        torch.save(dev_slots,os.path.join(output_folder, MEI.replace(" ","_")+"_dev_slots.pt"))    
    if len(test_X) > 0:
        torch.save(test_X,os.path.join(output_folder, MEI.replace(" ","_")+"_test_X.pt"))
        torch.save(test_y,os.path.join(output_folder, MEI.replace(" ","_")+"_test_y.pt"))
        torch.save(test_slots,os.path.join(output_folder, MEI.replace(" ","_")+"_test_slots.pt"))    
    print("\n\nMEI {}".format(MEI))
    print("Train has {} examples, dev has {}, and test has {}".format(len(train_X),len(dev_X), len(test_X)))
    print("[slots] Train has {} examples, dev has {}, and test has {}".format(len(train_slots),len(dev_slots), len(test_slots)))
    print("Skipped because of exceeding length {}, skipped for errors {}, skipped for modified=no {}".format(skipped_len, skipped_error, skipped_modified))
    print("\n\nDone.")


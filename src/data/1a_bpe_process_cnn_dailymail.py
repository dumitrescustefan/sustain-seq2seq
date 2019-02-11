import requests, os, sys, json, string
import tarfile
import unicodedata

import sentencepiece as spm

def process_tgz(tgz_file1, tgz_file2, dest_folder, size, spm_model):
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_model)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    tar1 = tarfile.open(tgz_file1, 'r')
    files1 = tar1.getnames()
    files1 = [x for x in files1 if x.endswith(".story")]
    print("Files in TAR file1: "+str(len(files1)))
    print()

    tar2 = tarfile.open(tgz_file2, 'r')
    files2 = tar2.getnames()
    files2 = [x for x in files2 if x.endswith(".story")]
    print("Files in TAR file1: "+str(len(files2)))
    print()
    
    files = files1+files2
    batch = []
    last_i = 0
    for i in range(len(files)):
        if i % 50 == 0:
            print("{} / {}\t({}) ...".format(i, len(files), i*100.0/float(len(files))))
    
        #f=tar.extractfile("./dailymail/stories/06588a8ab74f068ec61b89de9ca03a28f5ebd6f4.story")
        if i<len(files1):
            f=tar1.extractfile(files[i])
        else:
            f=tar2.extractfile(files[i])
        byte_content=f.readlines()

        # convert all byte arrays to strings
        content = [b.decode("utf-8").strip() for b in byte_content]
        elem = {}
        elem["file"] = files[i]
        x, y = process_individual_file(content)        
        elem["x"] = []
        elem["y"] = []
        for line in x:
            elem["x"].append(sp.EncodeAsIds(line))
        for line in y:
            elem["y"].append(sp.EncodeAsIds(line))
        
        batch.append(elem)
        
        if len(batch)>=size:
            filename = str(i-size+1)+"-"+str(i)+".json"
            filename = os.path.join(dest_folder, filename)
            #json.dump(batch, open(filename,"w",encoding="utf-8"), indent=4, sort_keys=True)
            json.dump(batch, open(filename,"w",encoding="utf-8"))
            batch = []
            last_i = i
        
    if len(batch)>0:
        filename = str(last_i+1)+"-"+str(len(files)-1)+".json"
        filename = os.path.join(dest_folder, filename)            
        #json.dump(batch, open(filename,"w",encoding="utf-8"), indent=4, sort_keys=True)
        json.dump(batch, open(filename,"w",encoding="utf-8"))
    
            
def process_individual_file(content): # content is an array of raw uft-8 strings (all lines)
    # search for "UPDATED:"
    for i in range(len(content)):
        if content[i].startswith("UPDATED:"):
            content = content[i+4:]
            break
        if content[i].startswith("Last updated at"):
            content = content[i+2:]
            break

    # search for "--" in first line
    index = content[0].find("--")
    if index>0:
        content[0] = content[0][index+3:]    
    #print(">>>>>>>>>>>>>>>>>>>>>>>>")
    #print(content[0])
    
    #print("\n\n\n------")
    #print(content)

    
    # normal process
    i = 0
    state = 0
    x = []
    y = []
    
    
    line = content[0].replace(u'\xa0', u' ')
    line = line.replace(u'\u00a320M', u'-')                    
    line = unicodedata.normalize('NFKD', line)#.encode('ascii','ignore')
    
    x.append(line)
    for i in range(1, len(content)): 
        if content[i] == "":
            continue
        
        line = content[i].replace(u'\xa0', u' ')
        line = line.replace(u'\u00a320M', u'-')                
        #print()
        #print(line)
        line = unicodedata.normalize('NFKD', line)#.encode('ascii','ignore')
        #print(line)
        #input("")
        #\u00a320
        
        if line == "@highlight":
            state = 1 # switch to writing to y

        if state == 0:
            if content[i-1] == "": # add to new sentence
                x.append(line)
            else: # concat to last sentence
                x[-1] = x[-1] + " " + line
        else:
            if line == "@highlight" or line == "":
                continue                
            if content[i-1] == "": # add to new sentence
                y.append(line)
            else: # concat to last sentence
                y[-1] = y[-1] + " " + line

    # after processing
    y = [elem.replace("NEW: ","").replace("New :","").strip() for elem in y]
    
    for i in range(len(y)):
        if y[i][-1] not in string.punctuation:
            y[i] += "."
    
    #print()
    #print(x)
    #print(y)
            
    if len(y) == 0 or len(x) == 0:
        print(content)
        print("\n x and y below:")
        print(x)
        print(y)    
        input("ERROR")
        
    return x, y
    
#model_name = "cnndm.1K.bpe.model"   
model_name = "cnndm.8K.bpe.model"   
process_tgz("cnn_stories.tgz", "dailymail_stories.tgz", os.path.join("bpe_processed",model_name), 1000, os.path.join("bpe_models", model_name))

model_name = "cnndm.32K.bpe.model"   
process_tgz("cnn_stories.tgz", "dailymail_stories.tgz", os.path.join("bpe_processed",model_name), 1000, os.path.join("bpe_models", model_name))


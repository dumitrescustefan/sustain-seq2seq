import requests, os, sys, json
import tarfile
import unicodedata
import re
import numpy as np
import time
import random
import string
import json
from datetime import datetime
import spacy

random.seed(42)
np.random.seed(42)

def while_replace(string):
    while '  ' in string:
        string = string.replace('  ', ' ')
    return string
    
def clean_string(string):
    cleaned = string.replace("’","'")    
    cleaned = cleaned.replace("\r","")
    cleaned = cleaned.replace("\n"," ")
    cleaned = cleaned.replace("\t"," ")
    #cleaned = ''.join(filter(lambda x: x in string.printable, cleaned)) #removeNonAscii(cleaned)    
    cleaned = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '', cleaned) # URL remove
    cleaned = re.sub(r'[^@]+@[^@]+\.[^@]+', '', cleaned) # EMAIL remove
    return while_replace(cleaned.strip()) 
    

def process_document (text, nlp_core):        
    cleaned_text = [clean_string(line) for line in text]
    
    # split in sentences
    spacy_sentences = []    
    for line in cleaned_text:  
        doc = nlp_core(line)     
        spacy_sentences += doc.sents
    
    # tokenized_original
    tokenized_original_sentences = []
    for sent in spacy_sentences:
        tokenized_original_sentences.append([tok.text for tok in sent])
    
    # entity annonimizerd
    entity_anonimized_sentences = [] 
    entity_dict = {}
    entity_index = 0
    for sent in spacy_sentences:        
        i=-1
        sentence = []
        while i<len(sent)-1:
            i+=1        
            token = sent[i]
            #print(str(i)+" "+token.text)
            
            if token.ent_iob_ == "B": # new entity found
                my_entity = token.text
                while i<len(sent)-1:
                    if sent[i+1].ent_iob_ == "I":
                        my_entity += " " +sent[i+1].text
                        i+=1
                    else:
                        break                
                #sentence.append("<"+token.ent_type_+">")
                
                # is it a new entity?
                new_entity = True
                if my_entity in entity_dict:                     
                    tag, index = entity_dict[my_entity]
                    if tag == token.ent_type_: # get entity from dict
                        my_tag = "@"+tag+str(index) 
                        new_entity = False
                
                if new_entity:
                    my_tag = "@"+token.ent_type_+str(entity_index)                    
                    entity_dict[my_entity] = (token.ent_type_, entity_index)
                    entity_index+=1
                
                sentence.append(my_tag.lower())
                continue                           
            sentence.append(token.text.lower())  
            
        entity_anonimized_sentences.append(sentence)
        
    return tokenized_original_sentences, entity_anonimized_sentences    



def process_folder(input_folder, dest_folder, size, nlp_core):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    import glob
    json_files = sorted(glob.glob(os.path.join(input_folder,"*.json")))
    print(json_files)
    for json_file in json_files:        
        batch = []
        print("Processing "+json_file)        
        js = json.load(open(json_file))
        
        for article in js:
            sys.stdout.write(".")
            sys.stdout.flush()
            tokenized_original_sentences, entity_anonimized_sentences = process_document(article["x"], nlp_core)
            article["x_tokenized_original_sentences"] = tokenized_original_sentences
            article["x_entity_anonimized_sentences"] = entity_anonimized_sentences
            tokenized_original_sentences, entity_anonimized_sentences = process_document(article["y"], nlp_core)
            article["y_tokenized_original_sentences"] = tokenized_original_sentences
            article["y_entity_anonimized_sentences"] = entity_anonimized_sentences
            batch.append(article)
            #break
        
        head, filename = os.path.split(json_file)        
        json.dump(batch, open(os.path.join(dest_folder, filename),"w",encoding="utf-8"), indent=4, sort_keys=True)
        #break
  
  
  
print("Loading spacy model ...")    
nlp_core = spacy.load('en_core_web_lg')
print("Done.")    
#text = ["John Cain, Santa Claus and Obama have 100 apples and went for 100 miles before stopping at McDonald's wednesday. Then John Cain said to Mary: it's 100 kilometers!","Yes, Mary said."]
#tokenized_original_sentences, sentences = process_document(text, nlp_core)
#print(tokenized_original_sentences)
#print()
#print(sentences)


process_folder(os.path.join("raw","cnn"), os.path.join("processed","text","cnn"), 1000, nlp_core)

process_folder(os.path.join("raw","dm"), os.path.join("processed","text","dm"), 1000, nlp_core)



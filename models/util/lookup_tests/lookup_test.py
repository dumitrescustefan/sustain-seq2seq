import os, sys
sys.path.append("..")
from lookup import Lookup

def test_lookup (lookup, text = "A test."):
    print(lookup)
    
    print("Testing with: [{}]".format(text))
    
    id_of_bos = lookup.convert_tokens_to_ids(lookup.bos_token)
    id_of_eos = lookup.convert_tokens_to_ids(lookup.eos_token)
    id_of_pad = lookup.convert_tokens_to_ids(lookup.pad_token)
    converted_bos_token = lookup.convert_ids_to_tokens(id_of_bos)
    converted_eos_token = lookup.convert_ids_to_tokens(id_of_eos)
    converted_pad_token = lookup.convert_ids_to_tokens(id_of_pad)
    
    print("bos_token {} = {} and converted back to token = {}".format(lookup.bos_token, id_of_bos, converted_bos_token))
    print("eos_token {} = {} and converted back to token = {}".format(lookup.eos_token, id_of_eos, converted_eos_token))
    print("pad_token {} = {} and converted back to token = {}".format(lookup.pad_token, id_of_pad, converted_pad_token))
        
    #print(lookup._tokenizer.all_special_ids)
    #print(lookup._tokenizer.all_special_tokens)
    #print(lookup._tokenizer.special_tokens_map)     
    
    print("\n0. Save/load lookup object:")
    if not os.path.exists(lookup.type):
        os.makedirs(lookup.type)
    lookup.save_special_tokens(file_prefix=os.path.join(lookup.type, lookup.type))
    lookup = Lookup(type=lookup.type) # recreate object
    lookup.load(file_prefix=os.path.join(lookup.type, lookup.type))
    print(lookup)
    
    print("\n1. String to tokens (tokenize):")
    tokens = lookup.tokenize(text)
    print(tokens)
    
    print("\n2. Tokens to ints (convert_tokens_to_ids):")
    ids = lookup.convert_tokens_to_ids(tokens)
    print(ids)
    
    print("\n2.5 Token to int (convert_tokens_to_ids with a single str):")
    id = lookup.convert_tokens_to_ids(tokens[0])
    print(id)
    
    print("\n3. Ints to tokens (convert_ids_to_tokens):")
    tokens = lookup.convert_ids_to_tokens(ids)
    print(tokens)
    
    print("\n3.5 Int to token (convert_ids_to_tokens with a single int):")
    token = lookup.convert_ids_to_tokens(id)
    print(token)
    
    print("\n4. Tokens to string (convert_tokens_to_string):")
    recreated_text = lookup.convert_tokens_to_string(tokens)
    print(recreated_text)
    
    print("\n5. String to ints (encode):")
    ids = lookup.encode(text)
    print(ids)
    
    print("\n6. Ints to string (decode):")
    recreated_text = lookup.decode(ids)
    print(recreated_text)
    
    print("\n7. Encode adding special tokens:")
    ids = lookup.encode(text, add_bos_eos_tokens=True)
    print(ids)
    print("How it looks like with tokens: {}".format(lookup.convert_ids_to_tokens(ids)))
        
    print("\n8. Decode skipping special tokens:")
    recreated_text = lookup.decode(ids, skip_bos_eos_tokens=True)
    print(recreated_text)
    
if __name__ == "__main__":
    # gpt2
    lookup = Lookup(type="gpt2")   
    test_lookup(lookup)    
   
    # bpe
    print("Create BPE model ...")
    lookup = Lookup(type="bpe")
    if not os.path.exists(lookup.type):
        os.makedirs(lookup.type)
        
    import sentencepiece as spm
    spm.SentencePieceTrainer.Train('--input=dummy_corpus.txt --model_prefix=bpe/bpe --character_coverage=1.0 --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --max_sentence_length=8000 --vocab_size=1024')
    
    lookup.load("bpe/bpe")
    test_lookup(lookup) 
    
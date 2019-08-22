import os, sys
from datetime import datetime

class Lookup():
    def __init__(self, type, folder = None):
        self.type = type
        
        self.bos_token = None
        self.eos_token = None
        self.unk_token = None
        self.sep_token = None
        self.pad_token = None
        self.cls_token = None
        self.mask_token = None
        
        self.w2i = {}
        self.i2w = {}
        
        if type == "gpt2":
            from pytorch_transformers import GPT2Tokenizer
            self._init_gpt2()
            
        if type == "bpe":
            import sentencepiece as spm
            self._init_bpe()
            
        if folder:
            self.load(folder)
        
                
    def _init_bpe(self, folder):
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'
        self.sep_token = '<SEP>'
        self.pad_token = '<PAD>'
        self.cls_token = '<CLS>'
        self.mask_token = '<MSK>'
        
        

            
    def _init_gpt2(self, folder):
        
        self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        for i in range(len(self._tokenizer)):
            token = self._tokenizer.convert_ids_to_tokens(i)
            self.i2w[i] = token
            self.w2i[token] = i 
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.unk_token = self.tokenizer.unk_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_token = self.tokenizer.pad_token
        self.cls_token = self.tokenizer.cls_token
        self.mask_token = self.tokenizer.mask_token
            
    def save(self, filename):
        pass
    
    def load(self, file):
        if self.type == "gpt2":
            pass
            
        if self.type == "bpe":
            self._tokenizer = spm.SentencePieceProcessor()
            self._tokenizer.Load(file+".model")
            index = -1  
            with open(file+".vocab","r",encoding="utf8") as f:    
                for line in f:
                    index+=1
                    word = line.split("\t")[0]
                    self.w2i[word] = index
                    self.i2w[str(index)] = word

          
    """
    convert_ids_to_tokens (self, tokens)
    
    convert_tokens_to_ids (self, tokens)
    
    decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)

    encode(text, text_pair=None, add_special_tokens=False)    
    """
    
    def __len__(self):
        return len(self.w2i)
        
if __name__ == "__main__":
    lookup = Lookup(type="gpt2")
    print(len(lookup))
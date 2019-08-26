import os, sys, json
from datetime import datetime

class Lookup():
    def __init__(self, type, file_prefix = None):
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
            self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            for i in range(len(self._tokenizer)):
                token = self._tokenizer.convert_ids_to_tokens(i)
                self.i2w[i] = token
                self.w2i[token] = i             
            if self._tokenizer._bos_token: # using _xxx_token instead of xxx_token to silence gpt2tokenizer not set errors
                self.bos_token = self._tokenizer.bos_token
            if self._tokenizer._eos_token:
                self.eos_token = self._tokenizer.eos_token
            if self._tokenizer._unk_token:                
                self.unk_token = self._tokenizer.unk_token
            if self._tokenizer._sep_token:
                self.sep_token = self._tokenizer.sep_token
            if self._tokenizer._pad_token:
                self.pad_token = self._tokenizer.pad_token
            if self._tokenizer._cls_token:
                self.cls_token = self._tokenizer.cls_token
            if self._tokenizer._mask_token:
                self.mask_token = self._tokenizer.mask_token            
                
        if file_prefix:
            self.load(file_prefix)
        
    def save_additional_tokens(self, file_prefix):
        if self.type == "gpt2":
            additional_tokens = {}
            if self.bos_token:
                additional_tokens['bos_token'] = self.bos_token
            if self.eos_token:
                additional_tokens['eos_token'] = self.eos_token
            if self.unk_token:
                additional_tokens['unk_token'] = self.unk_token
            if self.sep_token:
                additional_tokens['sep_token'] = self.sep_token
            if self.pad_token:
                additional_tokens['pad_token'] = self.pad_token
            if self.cls_token:
                additional_tokens['cls_token'] = self.cls_token
            if self.mask_token:
                additional_tokens['mask_token'] = self.mask_token            
            json.dump(additional_tokens, open(file_prefix+".additional_tokens","w",encoding="utf8"), indent=4, sort_keys=True)
            
        if self.type == "bpe":
            additional_tokens = {}
            additional_tokens['bos_token'] = self.bos_token
            additional_tokens['eos_token'] = self.eos_token
            additional_tokens['unk_token'] = self.unk_token
            additional_tokens['sep_token'] = self.sep_token
            additional_tokens['pad_token'] = self.pad_token
            additional_tokens['cls_token'] = self.cls_token
            additional_tokens['mask_token'] = self.mask_token            
            json.dump(additional_tokens, open(file_prefix+".additional_tokens","w",encoding="utf8"), indent=4, sort_keys=True)
        
    
    def load(self, file_prefix):
        if self.type == "gpt2":
            if os.path.exists(file_prefix+".additional_tokens"):
                additional_tokens = json.load(open(file_prefix+".additional_tokens","r",encoding="utf8"))            
                if 'bos_token' in additional_tokens:
                    self.bos_token = additional_tokens['bos_token']
                if 'eos_token' in additional_tokens:
                    self.eos_token = additional_tokens['eos_token']
                if 'unk_token' in additional_tokens:
                    self.unk_token = additional_tokens['unk_token']
                if 'sep_token' in additional_tokens:
                    self.sep_token = additional_tokens['sep_token']
                if 'pad_token' in additional_tokens:
                    self.pad_token = additional_tokens['pad_token']
                if 'cls_token' in additional_tokens:
                    self.cls_token = additional_tokens['cls_token']
                if 'mask_token' in additional_tokens:
                    self.mask_token = additional_tokens['mask_token']
                self._add_additional_tokens_to_w2i_and_i2w()
                self._tokenizer.add_special_tokens(additional_tokens)
                
        if self.type == "bpe":
            import sentencepiece as spm 
            # step 1, load SentencePieceProcessor
            self._tokenizer = spm.SentencePieceProcessor()
            self._tokenizer.Load(file_prefix+".model")
            # step 2, create base w2i and i2w
            index = -1  
            with open(file_prefix+".vocab","r",encoding="utf8") as f:    
                for line in f:
                    index+=1
                    word = line.split("\t")[0]
                    self.w2i[word] = index
                    self.i2w[str(index)] = word
            # step 3, load additional_tokens, if any, and add to vocabulary
            if os.path.exists(file_prefix+".additional_tokens"):
                additional_tokens = json.load(open(file_prefix+".additional_tokens","r",encoding="utf8"))
                self.bos_token = additional_tokens['bos_token']
                self.eos_token = additional_tokens['eos_token']
                self.unk_token = additional_tokens['unk_token']
                self.sep_token = additional_tokens['sep_token']
                self.pad_token = additional_tokens['pad_token']
                self.cls_token = additional_tokens['cls_token']
                self.mask_token = additional_tokens['mask_token']
                self._add_additional_tokens_to_w2i_and_i2w()                
                
    def _add_additional_tokens_to_w2i_and_i2w (self):
        if self.bos_token:
            self.w2i[self.bos_token] = len(self.w2i)
            self.i2w[str(len(self.i2w))] = self.bos_token
        if self.eos_token:
            self.w2i[self.eos_token] = len(self.w2i)
            self.i2w[str(len(self.i2w))] = self.eos_token
        if self.unk_token:
            self.w2i[self.unk_token] = len(self.w2i)
            self.i2w[str(len(self.i2w))] = self.unk_token
        if self.sep_token:
            self.w2i[self.sep_token] = len(self.w2i)
            self.i2w[str(len(self.i2w))] = self.sep_token
        if self.pad_token:
            self.w2i[self.pad_token] = len(self.w2i)
            self.i2w[str(len(self.i2w))] = self.pad_token
        if self.cls_token:
            self.w2i[self.cls_token] = len(self.w2i)
            self.i2w[str(len(self.i2w))] = self.cls_token
        if self.mask_token:
            self.w2i[self.mask_token] = len(self.w2i)
            self.i2w[str(len(self.i2w))] = self.mask_token            
                
    def tokenize (self, text):
        """ Converts a string in a sequence of tokens (string)
        """
        if self.type == "gpt2":
            return self._tokenizer.tokenize(text)
        if self.type == "bpe":
            return self._tokenizer.EncodeAsPieces(text)
        
    def convert_tokens_to_ids (self, tokens):
        if self.type == "gpt2":
            return self._tokenizer.convert_tokens_to_ids(tokens)
        if self.type == "bpe":
            if isinstance(tokens, list):
                return [self._tokenizer.PieceToId(token) for token in tokens] # [self.w2i[token] for token in tokens]                            
            elif isinstance(tokens, str):
                return self._tokenizer.PieceToId(tokens) # self.w2i[tokens]
            else:
                raise Exception("Lookup convert_tokens_to_ids error: token_ids is not str or list!")
                
    def convert_ids_to_tokens (self, token_ids):
        if self.type == "gpt2":
            return self._tokenizer.convert_ids_to_tokens(token_ids)
        if self.type == "bpe":
            if isinstance(token_ids, list):
                return [self._tokenizer.IdToPiece(id) for id in token_ids]  # [self.i2w[str(id)] for id in token_ids]            
            elif isinstance(token_ids, int):
                return self._tokenizer.IdToPiece(token_ids) # self.i2w[str(token_ids)]
            else:
                raise Exception("Lookup convert_ids_to_tokens error: token_ids is not int or list!")
                
    def convert_tokens_to_string (self, tokens):
        if self.type == "gpt2":
            return self._tokenizer.convert_tokens_to_string(tokens)
        if self.type == "bpe":
            if isinstance(tokens, list):
                return self._tokenizer.DecodePieces(tokens)
            elif isinstance(tokens, str):
                return self._tokenizer.DecodePieces([tokens])
            else:
                raise Exception("Lookup convert_tokens_to_string error: tokens is not str or list!")
    
    def encode (self, text, add_bos_eos_tokens=False):
        tokens = self.tokenize(text)
        if add_bos_eos_tokens:
            if not self.bos_token or not self.eos_token:
                raise Exception("Lookup encode error: {} model does not have BOS or EOS tokens set!")            
            return [self.w2i[self.bos_token]] + self.convert_tokens_to_ids(tokens) + [self.w2i[self.eos_token]]
        else:
            return self.convert_tokens_to_ids(tokens)
        
    def decode (self, token_ids, skip_bos_eos_tokens=False):                
        if skip_bos_eos_tokens:            
            if not self.bos_token or not self.eos_token:                
                raise Exception("Lookup decode error: {} model does not have BOS or EOS tokens set!")                                  
            if token_ids[0] == self.w2i[self.bos_token]:
                token_ids = token_ids[1:]
            if token_ids[-1] == self.w2i[self.eos_token]:
                token_ids = token_ids[:-1]        
        tokens = self.convert_ids_to_tokens(token_ids)                
        return self.convert_tokens_to_string(tokens)
    
    def __len__(self):
        return len(self.w2i)
        
if __name__ == "__main__":
    lookup = Lookup(type="gpt2")
    print(len(lookup))
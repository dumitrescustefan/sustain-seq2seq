import os, sys, json
from datetime import datetime

"""
    Create BPE lookup:
    1. create object (will init special tokens)
    2. create BPE model with sentencepiece.train
    3. lookup.load(path to model (prefix only, without .model)), this will shift special tokens up the vocab
    4. lookup.save_special_tokens() to save with new positions
    Load BPE lookup:
    1. create lookup object
    2. load()
    
    Create/load GPT2:
    1. create object
    2. load()
"""

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
                
        if type == "gpt2":
            from pytorch_transformers import GPT2Tokenizer 
            self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            # add <PAD> special token
            self._tokenizer.add_special_tokens({'pad_token': '<PAD>'})
                        
            for i in range(len(self._tokenizer)):
                token = self._tokenizer.convert_ids_to_tokens(i)                       
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
        
        if type == "bpe":
            self.bpe_vocab_size = 0
            self.bos_token = "<BOS>"
            self.eos_token = "<EOS>"
            self.unk_token = "<UNK>"
            self.sep_token = "<SEP>"
            self.pad_token = "<PAD>"
            self.cls_token = "<CLS>"
            self.mask_token = "<MASK>"
            self._recreate_special_tokens()            
        
        if file_prefix:
            self.load(file_prefix)
        
    def save_special_tokens(self, file_prefix):
        if self.type == "gpt2":
            special_tokens = {}
            if self.bos_token:
                special_tokens['bos_token'] = self.bos_token
            if self.eos_token:
                special_tokens['eos_token'] = self.eos_token
            if self.unk_token:
                special_tokens['unk_token'] = self.unk_token
            if self.sep_token:
                special_tokens['sep_token'] = self.sep_token
            if self.pad_token:
                special_tokens['pad_token'] = self.pad_token
            if self.cls_token:
                special_tokens['cls_token'] = self.cls_token
            if self.mask_token:
                special_tokens['mask_token'] = self.mask_token            
            json.dump(special_tokens, open(file_prefix+".special_tokens","w",encoding="utf8"), indent=4, sort_keys=True)            
            self._tokenizer.add_special_tokens(special_tokens) 
            
        if self.type == "bpe":
            obj = {}
            obj["special_token2id"] = self.special_token2id
            obj["special_id2token"] = self.special_id2token
            obj["special_token_map"] = self.special_token_map
            json.dump(obj, open(file_prefix+".special_tokens","w",encoding="utf8"), indent=4, sort_keys=True)
                        
            
    
    def load(self, file_prefix):
        if self.type == "gpt2":
            if os.path.exists(file_prefix+".special_tokens"):
                special_tokens = json.load(open(file_prefix+".special_tokens","r",encoding="utf8"))            
                if 'bos_token' in special_tokens:
                    self.bos_token = special_tokens['bos_token']
                if 'eos_token' in special_tokens:
                    self.eos_token = special_tokens['eos_token']
                if 'unk_token' in special_tokens:
                    self.unk_token = special_tokens['unk_token']
                if 'sep_token' in special_tokens:
                    self.sep_token = special_tokens['sep_token']
                if 'pad_token' in special_tokens:
                    self.pad_token = special_tokens['pad_token']
                if 'cls_token' in special_tokens:
                    self.cls_token = special_tokens['cls_token']
                if 'mask_token' in special_tokens:
                    self.mask_token = special_tokens['mask_token']
                self._tokenizer.add_special_tokens(special_tokens)                
                
        if self.type == "bpe":
            import sentencepiece as spm 
            # step 1, load SentencePieceProcessor
            self._tokenizer = spm.SentencePieceProcessor()
            self._tokenizer.Load(file_prefix+".model")   

            # step 2, get vocab size from .vocab
            with open(file_prefix+".vocab","r",encoding="utf8") as f:    
                all_lines = f.readlines()
            self.bpe_vocab_size = len(all_lines)            
            
            # step 3, load special_tokens, if any, and add to vocabulary
            if os.path.exists(file_prefix+".special_tokens"):
                obj = json.load(open(file_prefix+".special_tokens","r",encoding="utf8"))
                self.special_token2id = obj["special_token2id"]
                self.special_id2token = obj["special_id2token"]
                self.special_token_map = obj['special_token_map']
                self.bos_token = self.special_token_map.get('bos_token', None)
                self.eos_token = self.special_token_map.get('eos_token', None)
                self.unk_token = self.special_token_map.get('unk_token', None)
                self.sep_token = self.special_token_map.get('sep_token', None)
                self.pad_token = self.special_token_map.get('pad_token', None)
                self.cls_token = self.special_token_map.get('cls_token', None)
                self.mask_token = self.special_token_map.get('mask_token', None)
                self._recreate_special_tokens()       
                
    def _recreate_special_tokens (self):
        self.special_token2id = {}
        self.special_id2token = {}
        self.special_token_map = {}
        if self.pad_token:
            self.special_token2id[self.pad_token] = self.bpe_vocab_size+len(self.special_token2id)
            self.special_id2token[str(self.bpe_vocab_size+len(self.special_id2token))] = self.pad_token
            self.special_token_map['pad_token'] = self.pad_token
        if self.unk_token:
            self.special_token2id[self.unk_token] = self.bpe_vocab_size+len(self.special_token2id)
            self.special_id2token[str(self.bpe_vocab_size+len(self.special_id2token))] = self.unk_token
            self.special_token_map['unk_token'] = self.unk_token
        if self.bos_token:
            self.special_token2id[self.bos_token] = self.bpe_vocab_size+len(self.special_token2id)
            self.special_id2token[str(self.bpe_vocab_size+len(self.special_id2token))] = self.bos_token
            self.special_token_map['bos_token'] = self.bos_token
        if self.eos_token:
            self.special_token2id[self.eos_token] = self.bpe_vocab_size+len(self.special_token2id)
            self.special_id2token[str(self.bpe_vocab_size+len(self.special_id2token))] = self.eos_token 
            self.special_token_map['eos_token'] = self.eos_token            
        if self.sep_token:
            self.special_token2id[self.sep_token] = self.bpe_vocab_size+len(self.special_token2id)
            self.special_id2token[str(self.bpe_vocab_size+len(self.special_id2token))] = self.sep_token
            self.special_token_map['sep_token'] = self.sep_token            
        if self.cls_token:
            self.special_token2id[self.cls_token] = self.bpe_vocab_size+len(self.special_token2id)
            self.special_id2token[str(self.bpe_vocab_size+len(self.special_id2token))] = self.cls_token
            self.special_token_map['cls_token'] = self.cls_token
        if self.mask_token:
            self.special_token2id[self.mask_token] = self.bpe_vocab_size+len(self.special_token2id)
            self.special_id2token[str(self.bpe_vocab_size+len(self.special_id2token))] = self.mask_token
            self.special_token_map['mask_token'] = self.mask_token
                
    def tokenize (self, text):
        if self.type == "gpt2":
            return self._tokenizer.tokenize(text)
        if self.type == "bpe":
            return self._tokenizer.EncodeAsPieces(text)
        
    def convert_tokens_to_ids (self, tokens):
        if self.type == "gpt2":
            return self._tokenizer.convert_tokens_to_ids(tokens)
        if self.type == "bpe":
            if isinstance(tokens, list):
                return [self._PieceToId(token) for token in tokens]                          
            elif isinstance(tokens, str):
                return self._PieceToId(tokens) 
            else:
                raise Exception("Lookup convert_tokens_to_ids error: token_ids is not str or list!")
                
    def convert_ids_to_tokens (self, token_ids):
        if self.type == "gpt2":
            return self._tokenizer.convert_ids_to_tokens(token_ids)
        if self.type == "bpe":
            if isinstance(token_ids, list):
                return [self._IdToPiece(id) for id in token_ids]           
            elif isinstance(token_ids, int):
                return self._IdToPiece(token_ids) 
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
            return [self.convert_tokens_to_ids(self.bos_token)] + self.convert_tokens_to_ids(tokens) + [self.convert_tokens_to_ids(self.eos_token)]
        else:
            return self.convert_tokens_to_ids(tokens)
        
    def decode (self, token_ids, skip_bos_eos_tokens=False):                
        if skip_bos_eos_tokens:            
            if not self.bos_token or not self.eos_token:                
                raise Exception("Lookup decode error: {} model does not have BOS or EOS tokens set!")                                  
            if token_ids[0] == self.convert_tokens_to_ids(self.bos_token):
                token_ids = token_ids[1:]
            if token_ids[-1] == self.convert_tokens_to_ids(self.eos_token):
                token_ids = token_ids[:-1]        
        tokens = self.convert_ids_to_tokens(token_ids)                
        return self.convert_tokens_to_string(tokens)
    
    def _PieceToId(self, token): # just for bpe
        if token in self.special_token2id:
            return self.special_token2id[token]
        return self._tokenizer.PieceToId(token)
    
    def _IdToPiece(self, id): # just for bpe
        if str(id) in self.special_id2token:
            return self.special_id2token[str(id)]
        return self._tokenizer.IdToPiece(id)
    
    def __len__(self):
        if self.type == "gpt2":
            return len(self._tokenizer)
        if self.type == "bpe":
            return self.bpe_vocab_size+len(self.special_id2token)
            
    def __repr__(self):
        s = "Lookup(type={}, vocab_size={}):".format(self.type, len(self))
        s += "\nSpecial tokens: "
        if self.bos_token:
            s += "\n\t bos_token={}, id {}".format(self.bos_token, self.convert_tokens_to_ids(self.bos_token))
        if self.eos_token:
            s += "\n\t eos_token={}, id {}".format(self.eos_token, self.convert_tokens_to_ids(self.eos_token))
        if self.unk_token:
            s += "\n\t unk_token={}, id {}".format(self.unk_token, self.convert_tokens_to_ids(self.unk_token))
        if self.sep_token:
            s += "\n\t sep_token={}, id {}".format(self.sep_token, self.convert_tokens_to_ids(self.sep_token))
        if self.pad_token:
            s += "\n\t pad_token={}, id {}".format(self.pad_token, self.convert_tokens_to_ids(self.pad_token))
        if self.cls_token:
            s += "\n\t cls_token={}, id {}".format(self.cls_token, self.convert_tokens_to_ids(self.cls_token))
        if self.mask_token:
            s += "\n\t mask_token={}, id {}".format(self.mask_token, self.convert_tokens_to_ids(self.mask_token))
        return s
        
        
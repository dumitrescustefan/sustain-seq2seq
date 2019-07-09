import sys
sys.path.insert(0, '../..')

import dynet_config
# set random seed to have the same result each time
dynet_config.set(random_seed=0)
import dynet as dy
import numpy as np


class Encoder():
    def __init__(self, model, enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers, enc_lstm_dropout, enc_dropout):
        self.model = model
        self.train = False
        self.enc_dropout = enc_dropout
        self.enc_lstm_dropout = enc_lstm_dropout
        self.enc_num_layers = enc_num_layers
        self.embedding = self.model.add_lookup_parameters((enc_vocab_size, enc_emb_dim))
        self.rnn = dy.BiRNNBuilder(enc_num_layers, enc_emb_dim, enc_hidden_dim, self.model, dy.LSTMBuilder)
                
    
    def train (self):
        self.rnn.set_dropout(self.enc_lstm_dropout)
        self.train = True
    
    def train (self):
        self.rnn.set_dropout(0.)
        self.train = False   
        
    def forward(self, input):    
        # input is a sequence of numbers
        # output is a tensor of size [enc_seq_len, enc_hidden_dim], encoder states of size []
        
        #embeddings = self.embeddings(input)
        embeddings = [self.words_lookup[w] for w in sentence]
        if self.train:
            embeddings = dy.dropout(embeddings, self.enc_dropout)
        
        
        output, enc_states = lstm.transduce([dy.zeros(5, batch_size=2)])
        
        
        
class Decoder():
    def __init__(self, enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers, enc_lstm_dropout, enc_dropout):
        pass       
        

    def forward(self, input):    
        pass
        
class EncoderDecoder():
    def __init__(self, enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers, enc_lstm_dropout, enc_dropout):
        pass       
        

    def forward(self, input):    
        pass
"""
    def __init__(self,
                 # encoder params
                 enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers, enc_lstm_dropout, enc_dropout,
                 # decoder params
                 dec_input_dim, dec_emb_dim, dec_hidden_dim, dec_num_layers, dec_lstm_dropout, dec_dropout, dec_vocab_size, dec_attention_type, dec_transfer_hidden=False):
        
      
      
        super(LSTMEncoderDecoderWithAttention, self).__init__(        
            # encoder params
            LSTMEncoder, enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers, enc_lstm_dropout, enc_dropout,
            # decoder params
            LSTMDecoderWithAttention, dec_input_dim, dec_emb_dim, dec_hidden_dim, dec_num_layers, dec_vocab_size, dec_lstm_dropout, dec_dropout, dec_attention_type, dec_transfer_hidden)

        
        self.to(self.device)
"""
    
        
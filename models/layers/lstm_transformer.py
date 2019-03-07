import torch
import torch.nn as nn
import math 
import numpy as np
import random

from pprint import pprint

from input import InputLayerWithAbsolutePosition
from attention import SelfAttention

class SelfAttentionLSTMEncoderLayer(nn.Module):
    def __init__(self, input_dim, rnn_hidden_dim, rnn_layers, drop_prob=0.2, attention_probs_dropout_prob=0.2, num_attention_heads=8):        
        """
            This is a layer that takes as input a tensor (batch_size, seq_len, input_dim)
            passes it through the self attention that outputs (batch_size, seq_len, input_dim) (similar to input)
            and then runs a bidirectional RNN. It outputs a (batch_size, seq_len, rnn_hidden_dim*2) tensor.
        """
        super(SelfAttentionLSTMEncoderLayer, self).__init__()
        
        self.rnn_layers = rnn_layers
        self.rnn_hidden_dim = rnn_hidden_dim                     
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        
        self.self_attention = SelfAttention(input_dim, num_attention_heads, attention_probs_dropout_prob = attention_probs_dropout_prob)
        if rnn_layers == 1: # dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects rnn_layers greater than 1
            drop_prob = 0. 
        self.lstm = nn.LSTM(input_dim, rnn_hidden_dim, rnn_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop_prob)        
     
    def forward(self, input_tensor, attention_mask, rnn_hidden): # input tensor is (batch_size, max_seq_len, rnn_hidden_dim)
        batch_size = input_tensor.size(0)
        # input_tensor is (batch_size, seq_len, input_dim)
        self_attention_tensor = self.self_attention(input_tensor, attention_mask)
        # self_attention_tensor is (batch_size, seq_len, input_dim)
        
        lstm_output, rnn_hidden = self.lstm(self_attention_tensor, rnn_hidden)        
        # lstm_output is (batch_size, seq_len, rnn_hidden_dim * 2)
        
        lstm_output = self.dropout(lstm_output)        
        return lstm_output, rnn_hidden
    
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data # get any parameter of the network, use new to obtain same type of variable
        # weights are (n_layers*n_directions, batch_size, hidden_dim)
        hidd = weight.new(self.rnn_layers*2, batch_size, self.rnn_hidden_dim)
        cell = weight.new(self.rnn_layers*2, batch_size, self.rnn_hidden_dim)
                
        nn.init.xavier_normal_(hidd) # in-place xavier init        
        nn.init.xavier_normal_(cell) # in-place xavier init
        
        if torch.cuda.is_available():
            return ( hidd.cuda(), cell.cuda() )
        else:
            return ( hidd, cell )        

class SelfAttentionEncoderStack(nn.Module):
    """
    
    """
    def __init__(self, n_layers, input_dim, rnn_hidden_dim, max_seq_len=512, rnn_layers=1, drop_prob=0.2, attention_probs_dropout_prob=0.2, num_attention_heads = 8):
        super(SelfAttentionEncoderStack, self).__init__()
        
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.max_seq_len = max_seq_len
        self.rnn_layers = rnn_layers
        self.drop_prob = drop_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_attention_heads = num_attention_heads
        
        base_layer = SelfAttentionLSTMEncoderLayer(input_dim, rnn_hidden_dim, rnn_layers, drop_prob, attention_probs_dropout_prob, num_attention_heads)        
        if n_layers == 1:
            self.stack = nn.ModuleList([base_layer])
        else: 
            self.stack = nn.ModuleList([base_layer] + [SelfAttentionLSTMEncoderLayer(rnn_hidden_dim, rnn_hidden_dim, rnn_layers, drop_prob, attention_probs_dropout_prob, num_attention_heads) for _ in range(n_layers-1)])
    
    def forward(self, input_tensor, attention_mask = None, return_all_layers = True, return_all_states = True):    
        batch_size = input_tensor.size(0)
        """
            return_all_states returns a tensor of n_layers containing either the last state(concat of fw and bw) or all outputs for all timesteps
            return_all_layers returns either all n_layers or just the final layer's output
            if return_all_states is True:
                output is (batch_size, n_layers_or_last_1, seq_len, rnn_hidden_dim*2)
            else:
                output is (batch_size, n_layers_or_last_1, 1, rnn_hidden_dim*2)
        """
        # todo XXX if attention_mask is none, all are treated
        
        # forward        
        output = None
        for i, layer in enumerate(self.stack):            
            output_tensor, _ = layer(input_tensor, attention_mask, layer.init_hidden(batch_size))
            input_tensor = output_tensor
            # output_tensor is (batch_size, seq_len, rnn_hidden_dim * 2)
            
            # unsqueeze to (batch_size, 1, seq_len, rnn_hidden_dim * 2)
            output_tensor = output_tensor.unsqueeze(1)
                        
            if not return_all_states: # extract only last state, so generate a (batch_size, 1, 1, rnn_hidden_dim * 2) tensor
                output_tensor = output_tensor[:,:,self.max_seq_len-1:self.max_seq_len,:] # don't drop dim 3, so slice with range                
            # else, output_tensor remains (batch_size, 1, seq_len, rnn_hidden_dim * 2)            
            # so, at this point, output_tensor is a (batch_size, 1, 1-or-seq_len, rnn_hidden_dim * 2)
            
            if return_all_layers == True: # now, if we want all layers, we need to concat, otherwise just return last output_tensor as output
                if output == None: # first layer
                    output = output_tensor # (batch_size, 1, 1-or-seq_len, rnn_hidden_dim * 2)
                else: # concat along dim 1 towards (batch_size, n_layers, 1-or-seq_len, rnn_hidden_dim * 2)
                    output = torch.cat((output, output_tensor),dim=1)
            else:
                output = output_tensor # just return the (batch_size, 1, 1-or-seq_len, rnn_hidden_dim * 2)
        
        return output

        
        
#testing
if __name__ == '__main__':  
    net = SelfAttentionEncoderStack(n_layers=3, input_dim=128, rnn_hidden_dim=256)
    
    #encoder_outputs = torch.Tensor(64,399,512).float().random_(0,1)
    #decoder_hidden = torch.Tensor(64,1,256).float().random_(0,1)
    
    
    input_tensor = torch.Tensor(4, 10, 128).float().random_(0,1)
    attention_mask = torch.ones(4, 10)
    #attention_mask = torch.ones_like(input_ids)
        
    # We create a 3D attention mask from a 2D tensor mask.
    # Sizes are [batch_size, 1, 1, to_seq_length]
    # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
    # this attention mask is more simple than the triangular masking of causal attention
    # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    #extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    
    out = net(input_tensor, attention_mask = extended_attention_mask, return_all_layers = True, return_all_states = True)            
    print(out.size())
     
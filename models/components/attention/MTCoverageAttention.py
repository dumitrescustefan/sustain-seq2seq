import os, sys, math
sys.path.insert(0, '../../..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attention(nn.Module):
    def __init__(self, encoder_size, decoder_size, device, vocab_size = None):        
        super().__init__()
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size        
        self.vocab_size = vocab_size 
        
        # transforms encoder states into keys
        self.key_annotation_function = nn.Linear(self.encoder_size, self.encoder_size, bias=False)
        # transforms encoder states into values 
        self.value_annotation_function = nn.Linear(self.encoder_size, self.encoder_size, bias=False)
        # transforms the hidden state into query
        self.query_annotation_function = nn.Linear(self.decoder_size, self.encoder_size, bias=False) # NOTE: transforming q to K size 
        
        self.V = nn.Linear(self.encoder_size, 1, bias=False) 
        self.W1 = nn.Linear(self.encoder_size, self.encoder_size, bias=False)
        self.W2 = nn.Linear(self.encoder_size, self.encoder_size, bias=False) # encoder size because q is now K's size, otherwise dec_size to enc_size
        self.W3 = nn.Linear(self.vocab_size, self.encoder_size, bias=False)
        self.b = nn.Parameter(torch.zeros(self.encoder_size))
                        
        self.device = device
        self.to(self.device)

    def _reshape_state_h(self, state_h):    
        """
        Reshapes the hidden state to desired shape
        Input: [num_layers * 1, batch_size, decoder_hidden_size]  
        Output: [batch_size, 1, decoder_hidden_state]

        Args:
            state_h (tensor): Hidden state of the decoder.        
                [num_layers * 1, batch_size, decoder_hidden_size]
        
        Returns:
            The reshaped hidden state.
                [batch_size, 1, decoder_hidden_state]
        """
        num_layers, batch_size, hidden_size = state_h.size()
        # in case the decoder has more than 1 layer, take only the last one -> [1, batch_size, decoder_hidden_size]
        if num_layers > 1:
            state_h = state_h[num_layers-1:num_layers,:,:]

        # [1, batch_size, decoder_hidden_size] -> [batch_size, 1, decoder_hidden_size]
        return state_h.permute(1, 0, 2)
            
    def forward(self, enc_output, state_h, coverage, mask=None):
        """
        This function calculates the context vector of the attention layer, given the hidden state and the encoder
        last lstm layer output.

        Args:
            state_h (tensor): The raw hidden state of the decoder's LSTM 
                Shape: [num_layers * 1, batch_size, decoder_size].
            enc_output (tensor): The output of the last LSTM encoder layer. 
                Shape: [batch_size, seq_len, encoder_size].
            coverage(tensor): Coverage tensor
                Shape: [batch_size, vocab_size]
            mask (tensor): 1 and 0 as for encoder input
                Shape: [batch_size, seq_len].

        Returns:
            context (tensor): The context vector. Shape: [batch_size, encoder_size]
            attention_weights (tensor): Attention weights. Shape: [batch_size, seq_len, 1]
            coverage (tensor): The next coverage vector. Shape: [batch_size, vocab_size]
        """
        batch_size = enc_output.shape[0]
        seq_len = enc_output.shape[1]
        state_h = self._reshape_state_h(state_h) # [batch_size, 1, decoder_size]
        
        # get K, V, Q
        K = self.key_annotation_function(enc_output) # [batch_size, seq_len, encoder_size]
        V = self.value_annotation_function(enc_output) # [batch_size, seq_len, encoder_size]
        Q = self.query_annotation_function(state_h) # [batch_size, 1, encoder_size]
        
        # calculate energy        
        energy = self.V(torch.tanh(self.W1(K) + self.W2(Q) + self.W3(coverage.unsqueeze(1)) + self.b)) # [batch_size, seq_len, 1]        
        
        # mask with -inf paddings
        if mask is not None:            
            energy.masked_fill_(mask.unsqueeze(-1) == 0, -np.inf)
        
        # transform energy into probability distribution using softmax        
        attention_weights = torch.softmax(energy, dim=1) # [batch_size, seq_len, 1]
           
        # calculate weighted values z (element wise multiplication of energy * values)        
        # attention_weights is [batch_size, seq_len, 1], V is [batch_size, seq_len, encoder_size], z is same as V
        z = attention_weights*V # same as torch.mul(), element wise multiplication
        
        # finally, calculate context as the esum of z. 
        # z is [batch_size, seq_len, encoder_size], context will be [batch_size, encoder_size]
        context = torch.sum(z, dim=1)
        
        return context, attention_weights # [batch_size, encoder_size], [batch_size, seq_len, 1]


    
import os, sys
sys.path.insert(0, '../../..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.attention.Attention import Attention


class AdditiveAttention(Attention):
    def __init__(self, encoder_input_size, decoder_hidden_size, dropout, device):
        """
        Creates an Decoder with attention.

        Args:
            encoder_input_size (int): 
            decoder_hidden_size (int):             
            dropout (float): The dropout in the attention layer.
            device : The device to run the module on.
        """

        super(AdditiveAttention, self).__init__()

        self.attn1 = nn.Linear(encoder_input_size+decoder_hidden_size, int(encoder_input_size/2))
        self.attn2 = nn.Linear(int(encoder_input_size/2), 1)

        self.p_dropout = dropout

        self.to(device)

    def forward(self, state_h, enc_output):
        """
        This function calculates the context vector of the attention layer, given the hidden state and the encoder
        last lstm layer output.

        Args:
            state_h (tensor): The hidden state of the decoder's LSTM 
                Shape: [dec_num_layers * 1, batch_size, decoder_hidden_size].
            enc_output (tensor): The output of the last LSTM encoder layer. 
                Shape: [batch_size, seq_len, encoder_input_size].

        Returns:
            The context vector. Shape: [batch_size, encoder_input_size]
        """
        batch_size = enc_output.shape[0]
        seq_len = enc_output.shape[1]
        state_h = self.calculate_new_state_h(state_h, batch_size, seq_len)

        # Concatenates the encoder output with the new hidden state over the third dimension.
        attn_input = torch.cat((enc_output, state_h), dim=2)
        
        # Calculates the attention weights.
        attn_hidden = torch.tanh(F.dropout(self.attn1(attn_input), self.p_dropout))

        attn_weights = F.softmax(self.attn2(attn_hidden), dim=1)

        # Multiply the attention weights with the attn_weights.
        context_vector = torch.mul(enc_output, attn_weights)

        # Calculates the sum over the seq_len.
        context_vector = torch.sum(context_vector, dim=1) 
        
        return context_vector, attn_weights  # [batch_size, encoder_input_size], [batch_size, enc_seq_len, 1]
        
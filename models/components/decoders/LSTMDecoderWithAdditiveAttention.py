import os, sys
sys.path.insert(0, '../../..')

import torch
import torch.nn as nn

import numpy as np

from models.components.attention.AdditiveAttention import AdditiveAttention


class LSTMDecoderWithAdditiveAttention(nn.Module):
    def __init__(self, emb_dim, input_size, hidden_dim, num_layers, n_class, lstm_dropout, dropout, device):
        """
        Creates an Decoder with attention.

        Args:            
            emb_dim (int): Embeddings dimension.
            input_size (int): Size of input (E.g. size of encoder's output / hidden_dim*2 if bidirectional lstm, etc.)
            hidden_dim (int): LSTM hidden layers dimension.
            num_layers (int): Number of LSTM layers.
            n_class (int): Number of classes/ Vocabulary size.
            lstm_dropout (float): LSTM dropout.
            dropout (float): The dropout in the attention layer.
            device : The device to run the model on.
        """

        super(LSTMDecoderWithAdditiveAttention, self).__init__()

        self.embedding = nn.Embedding(n_class, emb_dim)
        self.attention = AdditiveAttention(encoder_input_size=input_size, decoder_hidden_size=hidden_dim, dropout=dropout, device=device)        
        self.lstm = nn.LSTM(emb_dim + input_size, hidden_dim, num_layers, dropout=lstm_dropout, batch_first=True)
        self.output_linear = nn.Linear(hidden_dim, n_class)
        
        self.to(device)

    def forward(self, input, enc_output, dec_states, teacher_forcing_ratio):
        """
        Args:
             input (tensor): The input of the decoder. 
                Shape: [batch_size, seq_len_dec].
             enc_output (tensor): The output of the encoder last LSTM layer. 
                Shape: [batch_size seq_len_enc, hidden_dim * 2].
             enc_states (tuple of tensors): The hidden and the cell states of the encoder last LSTM layer after being transformed by the linear layer. 
                State shape: [num_layers, batch_size, hidden_dim].

        Returns:
            The output of the decoder, a tensor that contains a sequence of tokens with the dimension equal to
            vocabulary size. Shape: [batch_size, seq_len_dec, n_class].
        """
        batch_size = input.shape[0]
        seq_len_dec = input.shape[1]        

        dec_states = (dec_states[0].contiguous(), dec_states[1].contiguous())

        # Calculates the context vector
        context_vector = self.attention(state_h=dec_states[0], enc_output = enc_output)

        # Concatenates the input of the <BOS> embedding with the context vector over the second dimensions. Transforms
        # the 2-D tensor to 3-D sequence tensor with length 1. [batch_size, emb_dim] + [batch_size, hidden_dim *
        # num_layers] -> [batch_size, 1, emb_dim + hidden_dim * num_layers].
        lstm_input = torch.cat((self.embedding(input[:, 0]), context_vector), dim=1).reshape(batch_size, 1, -1).contiguous()

        # Feeds the resulted first token to the lstm layer of the decoder. The initial state of the decoder is the
        # transformed state of the last LSTM layer of the encoder. [batch_size, seq_len_dec, hidden_dim],
        # [num_layers, batch_size, hidden_dim].
        curr_dec_output, curr_dec_states = self.lstm(lstm_input, dec_states)
        curr_output = self.output_linear(curr_dec_output)
        output = curr_output.clone()

        # Loop over the rest of tokens in the input seq_len_dec.
        for i in range(1, seq_len_dec):
            # Calculate the context vector at step i.
            context_vector = self.attention(state_h=curr_dec_states[0], enc_output=enc_output)

            if np.random.uniform(0, 1) < teacher_forcing_ratio:
                # Concatenates the i-th embedding of the input with the corresponding  context vector over the second
                # dimensions. Transforms the 2-D tensor to 3-D sequence tensor with length 1. [batch_size, emb_dim] +
                # [batch_size, hidden_dim * num_layers] -> [batch_size, 1, emb_dim + hidden_dim * num_layers].
                lstm_input = torch.cat((self.embedding(input[:, i]), context_vector), dim=1).reshape(batch_size, 1, -1)
            else:
                # Calculates the embeddings of the previous output. Counts the argmax over the last third dimension and
                # then squeezes the second dimension, the sequence length. [batch_size, emb_dim].
                prev_output_embeddings = self.embedding(torch.squeeze(torch.argmax(curr_output, dim=2), dim=1))

                # Concatenates the (i-1)-th embedding of the previous output with the corresponding  context vector over the second
                # dimensions. Transforms the 2-D tensor to 3-D sequence tensor with length 1. [batch_size, emb_dim] +
                # [batch_size, hidden_dim * num_layers] -> [batch_size, 1, emb_dim + hidden_dim * num_layers].
                lstm_input = torch.cat((prev_output_embeddings, context_vector), dim=1).reshape(batch_size, 1, -1)

            # Calculates the i-th decoder output and state. We initialize the decoder state with (i-1)-th state.
            # [batch_size, 1, hidden_dim], [num_layers, batch_size, hidden_dim].
            curr_dec_output, curr_dec_states = self.lstm(lstm_input, curr_dec_states)

            curr_output = self.output_linear(curr_dec_output)

            output = torch.cat((output, curr_output), dim=1)

        return output

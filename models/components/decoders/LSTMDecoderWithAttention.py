import sys
sys.path.insert(0, '../../..')

import torch

import numpy as np

from models.components.attention.Attention import Attention
from models.components.decoders.LSTMDecoder import LSTMDecoder


class LSTMDecoderWithAttention(LSTMDecoder):
    def __init__(self, emb_dim, input_size, hidden_dim, num_layers, n_class, lstm_dropout, dropout, attention_type, device):
        """
        Creates a Decoder with attention.

        Args :
            dropout (float): The dropout in the attention layer.

            see LSTMDecoder for further args info
        """

        super(LSTMDecoderWithAttention, self).__init__(emb_dim, input_size, hidden_dim, num_layers, n_class, lstm_dropout, dropout, device)

        self.attention = Attention(encoder_size=input_size, decoder_size=hidden_dim, device=device, type=attention_type)

        self.to(device)

    def forward(self, input, enc_output, dec_states, teacher_forcing_ratio):
        """
            See LSTMDecoder for further info.
        """
        batch_size = input.shape[0]
        seq_len_dec = input.shape[1]        
        attention_weights = []
        
        dec_states = (dec_states[0].contiguous(), dec_states[1].contiguous())
        output = torch.Tensor().to(self.device)
        output.requires_grad=False
        
        # Loop over the rest of tokens in the input seq_len_dec.
        for i in range(0, seq_len_dec-1):
            # Calculate the context vector at step i.
            # context_vector is [batch_size, encoder_size], attention_weights is [batch_size, seq_len, 1]
            context_vector, step_attention_weights = self.attention(state_h=dec_states[0], enc_output=enc_output)
            
            # save attention weights incrementally
            attention_weights.append(step_attention_weights.squeeze(2).cpu().tolist())
            
            if np.random.uniform(0, 1) < teacher_forcing_ratio or i is 0:
                # Concatenates the i-th embedding of the input with the corresponding  context vector over the second
                # dimensions. Transforms the 2-D tensor to 3-D sequence tensor with length 1. [batch_size, emb_dim] +
                # [batch_size, hidden_dim * num_layers] -> [batch_size, 1, emb_dim + hidden_dim * num_layers].                        
                lstm_input = torch.cat((self.dropout(self.embedding(input[:, i])), context_vector), dim=1).reshape(batch_size, 1, -1)
            else:
                # Calculates the embeddings of the previous output. Counts the argmax over the last third dimension and
                # then squeezes the second dimension, the sequence length. [batch_size, emb_dim].
                prev_output_embeddings = self.dropout(self.embedding(torch.squeeze(torch.argmax(lin_output, dim=2), dim=1)))
                
                # Concatenates the (i-1)-th embedding of the previous output with the corresponding  context vector over the second
                # dimensions. Transforms the 2-D tensor to 3-D sequence tensor with length 1. [batch_size, emb_dim] +
                # [batch_size, hidden_dim * num_layers] -> [batch_size, 1, emb_dim + hidden_dim * num_layers].
                lstm_input = torch.cat((prev_output_embeddings, context_vector), dim=1).reshape(batch_size, 1, -1)

            # Calculates the i-th decoder output and state. We initialize the decoder state with (i-1)-th state.
            # [batch_size, 1, hidden_dim], [num_layers, batch_size, hidden_dim].
            dec_output, dec_states = self.lstm(lstm_input, dec_states)

            # Maps the decoder output to the decoder vocab size space. 
            # [batch_size, 1, hidden_dim] -> [batch_size, 1, n_class].
            lin_output = self.output_linear(dec_output)

            # Adds the current output to the final output. [batch_size, i-1, n_class] -> [batch_size, i, n_class].
            output = torch.cat((output, lin_output), dim=1)


        # output is a tensor [batch_size, seq_len_dec, n_class]
        # attention_weights is a list of [batch_size, seq_len] elements, where each element is the softmax distribution for a timestep
        return output, attention_weights

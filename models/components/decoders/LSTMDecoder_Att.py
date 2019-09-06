import sys
sys.path.insert(0, '../../..')

import numpy as np

import torch
import torch.nn as nn

from models.components.attention.Attention import Attention

class Decoder(nn.Module):
    def __init__(self, emb_dim, input_size, hidden_dim, num_layers, vocab_size, lstm_dropout, dropout, attention_type, device):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.encoder_size = input_size
        self.decoder_size = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim + input_size, hidden_dim, num_layers, dropout=lstm_dropout, batch_first=True)
        self.attention = Attention(encoder_size=input_size, decoder_size=hidden_dim, device=device, type=attention_type)
        self.output_linear = nn.Linear(hidden_dim, vocab_size)
        
        # overwrite output to allow context from the attention to be added to the output layer
        intermediate_size = int( ((hidden_dim+input_size+emb_dim) + vocab_size) / 16 )
        self.output_linear = nn.Linear(hidden_dim+input_size+emb_dim, intermediate_size)
        self.softmax_linear = nn.Linear(intermediate_size, vocab_size)

        self.device = device
        self.to(device)

    def forward(self, x_tuple, y_tuple, enc_output, dec_states, teacher_forcing_ratio):
        """
            See LSTMDecoder for further info.
        """
        
        input, input_lengths = y_tuple[0], y_tuple[1]
        encoder_mask = x_tuple[2]
        
        batch_size = input.shape[0]
        seq_len_dec = input.shape[1]        
        attention_weights = []
        
        dec_states = (dec_states[0].contiguous(), dec_states[1].contiguous())
        output = torch.zeros(batch_size,seq_len_dec-1,self.vocab_size).to(self.device)
        output.requires_grad=False
        
        # Loop over the rest of tokens in the input seq_len_dec.
        for i in range(0, seq_len_dec-1):
            # Calculate the context vector at step i.
            # context_vector is [batch_size, encoder_size], attention_weights is [batch_size, seq_len, 1]
            context_vector, step_attention_weights = self.attention(state_h=dec_states[0], enc_output=enc_output, mask=encoder_mask)
            
            # save attention weights incrementally
            attention_weights.append(step_attention_weights.squeeze(2).cpu().tolist())
            
            if np.random.uniform(0, 1) < teacher_forcing_ratio or i is 0:
                # Concatenates the i-th embedding of the input with the corresponding  context vector over the second
                # dimensions. Transforms the 2-D tensor to 3-D sequence tensor with length 1. [batch_size, emb_dim] +
                # [batch_size, hidden_dim * num_layers] -> [batch_size, 1, emb_dim + hidden_dim * num_layers].                        
                prev_output_embeddings = self.dropout(self.embedding(input[:, i]))               
            else:
                # Calculates the embeddings of the previous output. Counts the argmax over the last third dimension and
                # then squeezes the second dimension, the sequence length. [batch_size, emb_dim].
                prev_output_embeddings = self.dropout(self.embedding(torch.squeeze(torch.argmax(softmax_output, dim=2), dim=1)))
                
            # Concatenates the (i-1)-th embedding of the previous output with the corresponding  context vector over the second
            # dimensions. Transforms the 2-D tensor to 3-D sequence tensor with length 1. [batch_size, emb_dim] +
            # [batch_size, hidden_dim * num_layers] -> [batch_size, 1, emb_dim + hidden_dim * num_layers].
            lstm_input = torch.cat((prev_output_embeddings, context_vector), dim=1).reshape(batch_size, 1, -1)

            # Calculates the i-th decoder output and state. We initialize the decoder state with (i-1)-th state.
            # [batch_size, 1, hidden_dim], [num_layers, batch_size, hidden_dim].
            dec_output, dec_states = self.lstm(lstm_input, dec_states)

            # Maps the decoder output to the decoder vocab size space. 
            # [batch_size, 1, hidden_dim + encoder_dim + emb_dim] -> [batch_size, 1, vocab_size].            
            lin_input = torch.cat( (dec_output, context_vector.unsqueeze(1), prev_output_embeddings.unsqueeze(1)) , dim = 2)
            lin_output = self.output_linear(lin_input) #lin_output = self.output_linear(dec_output)    
            softmax_output = self.softmax_linear(torch.tanh(lin_output))
            # Adds the current output to the final output. [batch_size, i-1, vocab_size] -> [batch_size, i, vocab_size].
            #output = torch.cat((output, lin_output), dim=1)            
            output[:,i,:] = softmax_output.squeeze(1)
            
        # output is a tensor [batch_size, seq_len_dec, vocab_size]
        # attention_weights is a list of [batch_size, seq_len] elements, where each element is the softmax distribution for a timestep
        return {'output':output, 'attention_weights':attention_weights}

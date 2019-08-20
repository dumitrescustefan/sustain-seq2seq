import sys
sys.path.insert(0, '../../..')

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.components.attention.SummaryCoverageAttention import Attention
from models.components.decoders.LSTMDecoder import LSTMDecoder


class LSTMDecoderWithAttentionAndPointerGenerator(LSTMDecoder):
    def __init__(self, emb_dim, input_size, hidden_dim, num_layers, n_class, lstm_dropout, dropout, attention_type, device):
        """
        Creates a Decoder with attention and Pointer network see https://nlp.stanford.edu/pubs/see2017get.pdf 

        Args :
            dropout (float): The dropout in the attention layer.

            see LSTMDecoder for further args info
        """

        super(LSTMDecoderWithAttentionAndPointerGenerator, self).__init__(emb_dim, input_size, hidden_dim, num_layers, n_class, lstm_dropout, dropout, device)
        
        self.emb_dim = emb_dim
        self.n_class = n_class
        self.encoder_size = input_size
        self.decoder_size = hidden_dim
        
        self.attention = Attention(encoder_size=input_size, decoder_size=hidden_dim, vocab_size=n_class, device=device, type=attention_type)

        # overwrite output to allow context from the attention to be added to the output layer
        self.output_linear = nn.Linear(hidden_dim+input_size+emb_dim, int((hidden_dim+input_size+emb_dim)/2))
        self.vocab_linear = nn.Linear(int((hidden_dim+input_size+emb_dim)/2), n_class)

        # p_gen parameters
        """
        self.context_linear = nn.Linear(self.encoder_size, self.decoder_size)
        self.decoder_state_linear = nn.Linear(self.decoder_size, self.decoder_size)
        self.decoder_input_linear = nn.Linear(self.emb_dim, self.decoder_size)
        self.p_gen_bias = nn.Parameter(torch.zeros(self.decoder_size))
        self.p_gen_linear = nn.Linear(self.decoder_size, 1)
        """
        self.p_gen_linear = nn.Linear(self.encoder_size + self.decoder_size*2 + self.emb_dim, 1)
        
        self.to(device)

    def forward(self, x_tuple, y_tuple, enc_output, dec_states, teacher_forcing_ratio):
        
        src, src_lengths, src_masks = x_tuple[0], x_tuple[1], x_tuple[2]
        tgt, tgt_lengths, tgt_masks = y_tuple[0], y_tuple[1], y_tuple[2]
        
        batch_size = tgt.shape[0]
        src_seq_len = src.shape[1]
        seq_len_dec = tgt.shape[1]        
        attention_weights = []
        
        dec_states = (dec_states[0].contiguous(), dec_states[1].contiguous())
        output = torch.zeros(batch_size,seq_len_dec-1,self.n_class).to(self.device)
        #output.requires_grad=False
        coverage = torch.zeros(batch_size, self.n_class).to(self.device)
        coverage_loss = 0
        
        # Loop over the rest of tokens in the tgt seq_len_dec.
        for i in range(0, seq_len_dec-1):
            # Calculate the context vector at step i.
            # context_vector is [batch_size, encoder_size], attention_weights is [batch_size, src_seq_len, 1], coverage is [batch_size, vocab_size]
            context_vector, step_attention_weights  = self.attention(state_h=dec_states[0], enc_output=enc_output, coverage=coverage, mask=src_masks)
            
            
            # save attention weights incrementally
            attention_weights.append(step_attention_weights.squeeze(2).cpu().tolist())
            
            if np.random.uniform(0, 1) < teacher_forcing_ratio or i is 0:
                # Concatenates the i-th embedding of the tgt with the corresponding  context vector over the second
                # dimensions. Transforms the 2-D tensor to 3-D sequence tensor with length 1. [batch_size, emb_dim] +
                # [batch_size, hidden_dim * num_layers] -> [batch_size, 1, emb_dim + hidden_dim * num_layers].                        
                prev_output_embeddings = self.dropout(self.embedding(tgt[:, i]))               
            else:
                # Calculates the embeddings of the previous output. Counts the argmax over the last third dimension and
                # then squeezes the second dimension, the sequence length. [batch_size, emb_dim].
                prev_output_embeddings = self.dropout(self.embedding(torch.squeeze(torch.argmax(vocab_logits, dim=2), dim=1)))
                
            # Concatenates the (i-1)-th embedding of the previous output with the corresponding  context vector over the second
            # dimensions. Transforms the 2-D tensor to 3-D sequence tensor with length 1. [batch_size, emb_dim] +
            # [batch_size, hidden_dim * num_layers] -> [batch_size, 1, emb_dim + hidden_dim * num_layers].
            lstm_input = torch.cat((prev_output_embeddings, context_vector), dim=1).reshape(batch_size, 1, -1)

            # Calculates the i-th decoder output and state. We initialize the decoder state with (i-1)-th state.
            # [batch_size, 1, hidden_dim], [num_layers, batch_size, hidden_dim].
            dec_output, dec_states = self.lstm(lstm_input, dec_states)

            # Maps the decoder output to the decoder vocab size space. 
            # [batch_size, 1, hidden_dim + encoder_dim + emb_dim] -> [batch_size, 1, n_class].            
            lin_input = torch.cat( (dec_output, context_vector.unsqueeze(1), prev_output_embeddings.unsqueeze(1)) , dim = 2)
            lin_output = self.output_linear(lin_input) #lin_output = self.output_linear(dec_output)    
            
            # vocab_dist is the softmaxed dist of the output of the generator, and is [batch_size, n_class]
            vocab_logits = self.vocab_linear(torch.tanh(lin_output))
            vocab_dist = torch.softmax(vocab_logits.squeeze(1), dim=1)
            
            # Calculate p_gen -> [batch_size, 1]
            """p_gen = self.context_linear(context_vector) + \
                self.decoder_state_linear(dec_states[-1][1]) + \
                self.decoder_input_linear(prev_output_embeddings) + \
                self.p_gen_bias
            p_gen = torch.sigmoid(self.p_gen_linear(p_gen)) 
            """
            # context_vector is [batch_size, encoder_size]
            # dec_states[-1][1] is [batch_size, decoder_size]
            # prev_output_embeddings is [batch_size, emb_dim]
            #print()
            #print(context_vector.size())
            #print(dec_states[-1][1].size())
            #print(prev_output_embeddings.size())
            p_gen_input = torch.cat( (context_vector, dec_states[-1][0], dec_states[-1][1], prev_output_embeddings) , dim = 1)
            p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input)) 
            
            # Calculate final distribution, final_dist will be [batch_size, n_class]
            # vocab_dist is [batch_size, n_class], step_attention_weights is [batch_size, src_seq_len, 1], src is [batch_size, src_seq_len] and contains indices
            # first, we must use step_attention_weights to get attention_dist to be [batch_size, n_class]
            attention_dist = torch.zeros(batch_size, self.n_class).to(self.device)
            attention_dist = attention_dist.scatter_add(1, src, step_attention_weights.squeeze(2))
            #print("Step {}, p_gen is {}, y is {}, generated: {}".format(i, p_gen[0].item(), tgt[0, i].item(), torch.argmax(vocab_logits, dim=2)[0].item()))
            final_dist = p_gen * vocab_dist + (1-p_gen) * attention_dist
            
            # Adds the current output to the final output. 
            #output = torch.cat((output, lin_output), dim=1)            
            output[:,i,:] = final_dist #softmax_output.squeeze(1)
            
            # for coverage only, calculate the next coverage by adding step_attention_weights where appropriate
            coverage = coverage.scatter_add(1, src, step_attention_weights.squeeze(2))
         
            # update coverage loss, both are [batch_size, n_class]
            coverage_loss = coverage_loss + torch.sum(torch.min(attention_dist, coverage))/batch_size
            
        # output is a tensor [batch_size, seq_len_dec, n_class]
        # attention_weights is a list of [batch_size, seq_len] elements, where each element is the softmax distribution for a timestep
        # coverage_loss is a scalar tensor
        return output, attention_weights, 0.#coverage_loss

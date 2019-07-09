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
        self.enc_dropout = enc_dropout
        self.enc_lstm_dropout = enc_lstm_dropout
        self.enc_num_layers = enc_num_layers
        # layers
        self.embedding = self.model.add_lookup_parameters((enc_vocab_size, enc_emb_dim))
        self.rnn = dy.BiRNNBuilder(enc_num_layers, enc_emb_dim, enc_hidden_dim, self.model, dy.LSTMBuilder)
        # other initializations
        self._train() # default: train mode
        
    def _train (self):
        self.rnn.set_dropout(self.enc_lstm_dropout)
        self.train = True
    
    def _eval (self):
        self.rnn.disable_dropout()
        self.train = False   
        
    def forward(self, input):    
        # input is a sequence of numbers (a list of ints)
        # output is a tensor of size [enc_seq_len, enc_hidden_dim^expr]
        embeddings = [self.embedding[x] for x in input]
        
        if self.train:
            embeddings = [dy.dropout(x, self.enc_dropout) for x in embeddings]
      
        output = self.rnn.transduce(embeddings) # TODO state resets at each transduce call??
        
        return output
        
        
class Decoder():
    def __init__(self, model, dec_emb_dim, dec_input_size, dec_hidden_dim, dec_num_layers, dec_vocab_size, dec_lstm_dropout, dec_dropout, attention_type):
        self.model = model       
        self.dec_emb_dim = dec_emb_dim
        self.dec_input_size = dec_input_size
        self.dec_hidden_dim = dec_hidden_dim
        self.dec_vocab_size = dec_vocab_size
        self.dec_lstm_dropout = dec_lstm_dropout
        self.dec_dropout = dec_dropout
        self.attention_type = attention_type
        # layers
        self.embedding = self.model.add_lookup_parameters((dec_vocab_size, dec_emb_dim))
        
        # other initializations
        self._train()
    
    def _train (self):
        self.rnn.set_dropout(self.enc_lstm_dropout)
        self.train = True
    
    def _eval (self):
        self.rnn.disable_dropout()
        self.train = False  
    
    def _attention(self, state_h, enc_output):
    
        return context_vector, step_attention_weights
    
    def forward(self, input, enc_output, dec_states, teacher_forcing_ratio): 
        seq_len = len(input)
        # input is a list of ints, starting with 2 "[BOS]"
        for i in range(0, seq_len-1):
            # Calculate the context vector at step i.
            # context_vector is [batch_size, encoder_size], attention_weights is [batch_size, seq_len, 1]
            context_vector, step_attention_weights = self._attention(state_h=dec_states[0], enc_output=enc_output)
            
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
            #output = torch.cat((output, lin_output), dim=1)            
            output[:,i,:] = lin_output.squeeze(1)
            
        # output is a tensor [batch_size, seq_len_dec, n_class]
        # attention_weights is a list of [batch_size, seq_len] elements, where each element is the softmax distribution for a timestep
        return output, attention_weights
        
        
        
        
class EncoderDecoder():
    def __init__(self, 
                model, 
                enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers, enc_lstm_dropout, enc_dropout,
                dec_emb_dim, dec_input_size, dec_hidden_dim, dec_num_layers, dec_vocab_size, dec_lstm_dropout, dec_dropout, attention_type):
        
        self.encoder = Encoder(model, enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers, enc_lstm_dropout, enc_dropout) 
        self.decoder = Decoder(model, dec_emb_dim, dec_input_size, dec_hidden_dim, dec_num_layers, dec_vocab_size, dec_lstm_dropout, dec_dropout, attention_type)
        self.train()
        
    def train (self):
        self.encoder._train()
        self.decoder._train()
        self.train = True
    
    def eval (self):
        self.encoder._eval()
        self.decoder._eval()
        self.train = False  
    
    def _transfer_hidden_from_encoder_to_decoder(self, enc_states):
        pass
        
    def forward(self, x, y, teacher_forcing_ratio=0.):    
        enc_output, enc_states = self.encoder.forward(x)
        
         # Calculates the output of the decoder.
        output, attention_weights = self.decoder.forward(y, enc_output, dec_states, teacher_forcing_ratio)

        # Creates a BOS tensor that must be added to the beginning of the output. [batch_size, 1, dec_vocab_size]
        bos_tensor = torch.zeros(batch_size, 1, self.dec_vocab_size).to(self.device)
        # Marks the corresponding BOS position with a probability of 1.
        bos_tensor[:, :, 2] = 1

        # Concatenates the BOS tensor with the output. [batch_size, dec_seq_len-1, dec_vocab_size] -> [batch_size,
        # dec_seq_len, dec_vocab_size]
        output = torch.cat((bos_tensor, output), dim=1)

        return output, attention_weights

    
        
if __name__ == "__main__":
    input = [2,4,5,6,3]
    enc = Encoder(model=dy.Model(), enc_vocab_size=10, enc_emb_dim=8, enc_hidden_dim=6, enc_num_layers=3, enc_lstm_dropout=0.4, enc_dropout=0.3)
    print(out)
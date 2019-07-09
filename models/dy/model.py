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
    def __init__(self, model, dec_emb_dim, enc_output_size, dec_hidden_dim, dec_num_layers, dec_vocab_size, dec_lstm_dropout, dec_dropout, attention_type):
        self.model = model       
        self.dec_emb_dim = dec_emb_dim
        self.enc_output_size = enc_output_size
        self.dec_hidden_dim = dec_hidden_dim
        self.dec_vocab_size = dec_vocab_size
        self.dec_lstm_dropout = dec_lstm_dropout
        self.dec_dropout = dec_dropout
        self.attention_type = attention_type
        # layers
        self.embedding = self.model.add_lookup_parameters((dec_vocab_size, dec_emb_dim))
        self.rnn = dy.VanillaLSTMBuilder(dec_num_layers, dec_emb_dim+enc_output_size, dec_hidden_dim, model)
        self.output_linear_W = self.model.add_parameters((enc_output_size, dec_hidden_dim))
        self.output_linear_b = self.model.add_parameters(enc_output_size)
        
        self.att_w1 = self.model.add_parameters((enc_output_size, enc_output_size))
        self.att_w2 = self.model.add_parameters((enc_output_size, enc_output_size))
        self.att_v = self.model.add_parameters((1, enc_output_size))

        # other initializations
        self._train()
    
    def _train (self):
        self.rnn.set_dropout(self.enc_lstm_dropout)
        self.train = True
    
    def _eval (self):
        self.rnn.disable_dropout()
        self.train = False  
    
    def _attention(self, dec_state, enc_output):
        w1 = self.att_w1.expr(update=True)
        w2 = self.att_w2.expr(update=True)
        v = self.att_v.expr(update=True)
        attention_weights = []

        w2dt = w2 * dec_state #dy.concatenate([state_fw.s()[-1], state_bw.s()[-1]])
        for input_vector in enc_output:
            attention_weight = v * dy.tanh(w1 * input_vector + w2dt)
            attention_weights.append(attention_weight)

        attention_weights = dy.softmax(dy.concatenate(attention_weights))

        context = dy.esum(
            [vector * attention_weight for vector, attention_weight in zip(enc_output, attention_weights)])
        
        step_attention_weights = None
        return context, step_attention_weights
        
    def forward(self, input, enc_output, teacher_forcing_ratio):         
        seq_len = len(input)
        output = []
        rnn = self.rnn.initial_state()
         
         attention = self._attend(rnn_outputs, rnn_states_fw[-1], rnn_states_bw[-1])

        # input is a list of ints, starting with 2 "[BOS]"
        for i in range(0, seq_len-1): # we stop when we feed the decoder the [EOS] and we take its output (thus the -1)
            # calculate the context vector at step i.
            # context_vector is [encoder_size], attention_weights is [seq_len] # todo
            context_vector, step_attention_weights = self._attention(dec_state=rnn.s(), enc_output=enc_output)
            
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
            rnn=rnn.add_input(lstm_input)
            dec_output = rnn.output()
            
            # Maps the decoder output to the decoder vocab size space. 
            lin_output = self.output_linear_W * dec_output + self.output_linear_b       

            # Adds the current output to the final output. [batch_size, i-1, n_class] -> [batch_size, i, n_class].
            #output = torch.cat((output, lin_output), dim=1)            
            output.append(lin_output)
            
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
        # x and y is a list of ints with BOS and EOS added
        
        enc_output = self.encoder.forward(x)
        
        output, attention_weights = self.decoder.forward(y, enc_output, teacher_forcing_ratio)

        return output, attention_weights

    
        
if __name__ == "__main__":
    input = [2,4,5,6,3]
    enc = Encoder(model=dy.Model(), enc_vocab_size=10, enc_emb_dim=8, enc_hidden_dim=6, enc_num_layers=3, enc_lstm_dropout=0.4, enc_dropout=0.3)
    print(out)
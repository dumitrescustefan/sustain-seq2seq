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
        self.rnn.set_dropout(self.dec_lstm_dropout)
        self.train = True
    
    def _eval (self):
        self.rnn.disable_dropout()
        self.train = False  
    
    def _attention(self, dec_state, enc_output):
        w1 = self.att_w1.expr(update=True)
        w2 = self.att_w2.expr(update=True)
        v = self.att_v.expr(update=True)
        attention_weights = []
        print("att")
        print(w2)
        print(dec_state)
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
        attention_weights = []
        rnn = self.rnn.initial_state([dy.inputVector(np.zeros(self.dec_hidden_dim)) for i in range(4)])
        #print([x.value() for x in rnn.s()])
        #print([x.value() for x in self.rnn.s()])
        print("Start forward loop:")
        #context, _ = self._attention(rnn.s(), enc_output)
        context = enc_output[-1]
        # input is a list of ints, starting with 2 "[BOS]"
        for i in range(0, seq_len-1): # we stop when we feed the decoder the [EOS] and we take its output (thus the -1)
            # calculate the context vector at step i.
            # context is [encoder_size], attention_weights is [seq_len] # todo
            #context, step_attention_weights = self._attention(rnn.s(), enc_output)
            context = enc_output[-1]
            step_attention_weights = []
            # save attention weights incrementally
            attention_weights.append(step_attention_weights)
            
            if np.random.uniform(0, 1) < teacher_forcing_ratio or i is 0:
                word_embedding = dy.dropout(self.embedding[input[i]], self.dec_dropout)                                
            else:
                prev_predicted_word_index = np.argmax(lin_output.value())                
                word_embedding = dy.dropout(self.embedding[prev_predicted_word_index], self.dec_dropout) 
                
            lstm_input = dy.concatenate([word_embedding, context])
            
            rnn=rnn.add_input(lstm_input) #cine e S si cum accesez state-ul curent, depinde de nr de layere?
            print([x.value() for x in rnn.s()])
            dec_output = rnn.output()
            
            # Maps the decoder output to the decoder vocab size space. 
            lin_output = self.output_linear_W * dec_output + self.output_linear_b       

            # Adds the current output to the final output. [batch_size, i-1, n_class] -> [batch_size, i, n_class].
            #output = torch.cat((output, lin_output), dim=1)            
            output.append(lin_output)
            print("Step {} predicted index = {}".format(i,np.argmax(lin_output.value())))
            
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
    model=dy.Model()
    dy.renew_cg()
    
    print("Encoder:")
    input = [2,4,5,6,3]
    enc = Encoder(model, enc_vocab_size=10, enc_emb_dim=8, enc_hidden_dim=6, enc_num_layers=3, enc_lstm_dropout=0.4, enc_dropout=0.3)
    enc_output = enc.forward(input)
    print("Encoder output:")
    print(enc_output)
    
    print("Decoder:")
    dec = Decoder(model, dec_emb_dim=4, enc_output_size=6, dec_hidden_dim=3, dec_num_layers=2, dec_vocab_size=10, dec_lstm_dropout=0.2, dec_dropout=0.2, attention_type="additive")
    dec_output = dec.forward(input, enc_output, 0.)
    print("Decoder output:")
    print(dec_output)
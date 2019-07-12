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
        self.training = True
    
    def _eval (self):
        self.rnn.disable_dropout()
        self.training = False   
        
    def forward(self, input):    
        # input is a sequence of numbers (a list of ints)
        # output is a tensor of size [enc_seq_len, enc_hidden_dim^expr]
        embeddings = [self.embedding[x] for x in input]
        
        if self.training:
            embeddings = [dy.dropout(x, self.enc_dropout) for x in embeddings]
      
        output = self.rnn.transduce(embeddings) 
        
        return output
        
        
class Decoder():
    def __init__(self, model, dec_emb_dim, enc_output_size, dec_hidden_dim, dec_num_layers, dec_vocab_size, dec_lstm_dropout, dec_dropout, attention_type):
        self.model = model       
        self.dec_emb_dim = dec_emb_dim
        self.enc_output_size = enc_output_size
        self.dec_hidden_dim = dec_hidden_dim
        self.dec_num_layers = dec_num_layers
        self.dec_vocab_size = dec_vocab_size
        self.dec_lstm_dropout = dec_lstm_dropout
        self.dec_dropout = dec_dropout
        self.attention_type = attention_type
        # layers
        self.embedding = self.model.add_lookup_parameters((dec_vocab_size, dec_emb_dim))
        self.rnn = dy.VanillaLSTMBuilder(dec_num_layers, dec_emb_dim+enc_output_size, dec_hidden_dim, model)
        self.output_linear_W = self.model.add_parameters((dec_vocab_size, dec_hidden_dim))
        self.output_linear_b = self.model.add_parameters(dec_vocab_size)
        
        self.att_w1 = self.model.add_parameters((enc_output_size, enc_output_size))
        self.att_w2 = self.model.add_parameters((enc_output_size, dec_hidden_dim))
        self.att_v = self.model.add_parameters((1, enc_output_size))
        
        # other initializations
        self._train()
    
    def _train (self):
        self.rnn.set_dropout(self.dec_lstm_dropout)
        self.training = True
    
    def _eval (self):
        self.rnn.disable_dropout()
        self.training = False  
    
    def _attention(self, dec_state, enc_output):
        # decoder_state is an array of 2*num_layers like c0, h0, .. cn, hn
        # take from it the last layer's hidden/output
        w1 = self.att_w1.expr(update=True)
        w2 = self.att_w2.expr(update=True)
        v = self.att_v.expr(update=True)
        attention_weights = []
        
        w2dt = w2 * dec_state[-1] 
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
        bos_vector = [0.] * self.dec_vocab_size
        bos_vector[2] = 1.        
        output = [dy.inputVector(bos_vector)]
        attention_weights = []
        rnn = self.rnn.initial_state([dy.inputVector(np.zeros(self.dec_hidden_dim)) for i in range(2*self.dec_num_layers)])
        #print("Start forward loop:")
        context, _ = self._attention(rnn.s(), enc_output)
        #context = enc_output[-1]
        # input is a list of ints, starting with 2 "[BOS]" 2 4 5 3
        for i in range(0, seq_len-1): # we stop when we feed the decoder the [EOS] and we take its output (thus the -1)
            # calculate the context vector at step i.
            # context is [encoder_size], attention_weights is [seq_len] # todo
            context, step_attention_weights = self._attention(rnn.s(), enc_output)
            #context = enc_output[-1]
            step_attention_weights = []
            # save attention weights incrementally
            #attention_weights.append(step_attention_weights)
            
            #if np.random.uniform(0, 1) < teacher_forcing_ratio or i is 0:
            word_embedding = dy.dropout(self.embedding[input[i]], self.dec_dropout)                                
            """else:
                #prev_predicted_word_index = np.argmax(lin_output.value()) 
                #index_vector = dy.inputVector(np.arange(self.dec_vocab_size))
                argmax = dy.argmax(lin_output, gradient_mode='zero_gradient')
                
                prev_embedding = dy.dropout(self.embedding*argmax, self.dec_dropout) 
                #prev_predicted_word_index = dy.sum_elems(dy.cmult(index_vector,dy.argmax(lin_output, gradient_mode='zero_gradient')))
                #print(prev_predicted_word_index.value())
                #word_embedding = dy.dropout(self.embedding[prev_predicted_word_index], self.dec_dropout) 
            """     
            lstm_input = dy.concatenate([word_embedding, context])
            
            rnn=rnn.add_input(lstm_input) 
            
            #print("rnn.s has {} vectors of length {}".format(len(rnn.s()), len(rnn.s()[0].value())))
            dec_output = rnn.output()
            
            # Maps the decoder output to the decoder vocab size space. 
            lin_output = self.output_linear_W.expr(update=True) * dec_output + self.output_linear_b.expr(update=True) 

            output.append(lin_output)
            #print("Step {} predicted index = {}".format(i,np.argmax(lin_output.value())))
        
        return output, attention_weights
        
        
        
class EncoderDecoder():
    def __init__(self, 
                model, 
                enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers, enc_lstm_dropout, enc_dropout,
                dec_emb_dim, dec_hidden_dim, dec_num_layers, dec_vocab_size, dec_lstm_dropout, dec_dropout, attention_type):
        
        self.model = model
        self.encoder = Encoder(model, enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers, enc_lstm_dropout, enc_dropout) 
        self.decoder = Decoder(model, dec_emb_dim, enc_hidden_dim, dec_hidden_dim, dec_num_layers, dec_vocab_size, dec_lstm_dropout, dec_dropout, attention_type)
        self.train()
        
    def train (self):
        self.encoder._train()
        self.decoder._train()
        self.training = True
    
    def eval (self):
        self.encoder._eval()
        self.decoder._eval()
        self.training = False  
    
    def _transfer_hidden_from_encoder_to_decoder(self, enc_states):
        pass
        
    def forward(self, x, y, teacher_forcing_ratio=0.):
        # x and y is a list of ints with BOS and EOS added
        
        enc_output = self.encoder.forward(x)
        
        output, attention_weights = self.decoder.forward(y, enc_output, teacher_forcing_ratio)

        return output, attention_weights # output is a list of logits

    
        
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
    dec = Decoder(model, dec_emb_dim=4, enc_output_size=6, dec_hidden_dim=3, dec_num_layers=3, dec_vocab_size=10, dec_lstm_dropout=0.2, dec_dropout=0.2, attention_type="additive")
    dec_output = dec.forward(input, enc_output, 0.)
    print("Decoder output:")
    print(dec_output)
    
    print("EncoderDecoder:")
    ed = EncoderDecoder(model, enc_vocab_size=10, enc_emb_dim=8, enc_hidden_dim=6, enc_num_layers=3, enc_lstm_dropout=0.4, enc_dropout=0.3, 
                    dec_emb_dim=4, dec_hidden_dim=3, dec_num_layers=3, dec_vocab_size=10, dec_lstm_dropout=0.2, dec_dropout=0.2, attention_type="additive")
    output = ed.forward(input, input, 0.)
    print("EncoderDecoder output:")
    print(output)
    
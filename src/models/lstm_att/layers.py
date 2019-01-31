import torch
import torch.nn as nn

from pprint import pprint

class SimpleLSTMEncoderLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.3):        
        super(SimpleLSTMEncoderLayer, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.train_on_gpu=torch.cuda.is_available()
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
     
    def forward(self, x, hidden):        
        batch_size = x.size(0)
        #64 x 399 
        x = self.embedding(x)
        #64 x 399 x embedding_dim
        
        # embeddings and lstm_out        
        lstm_out, hidden = self.lstm(x, hidden)        
        # lstm_out is 64 x 399 x hidden_dim * num_directions
        
        # from documentation, separate directions: the directions can be separated using output.view(seq_len, batch, num_directions, hidden_size), with forward and backward being direction 0 and 1 respectively. 
        # we'll use a larger hidden size in the decoder instead of projecting or summing the fw/bw hidden outputs

        return lstm_out, hidden
    
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data # get any parameter of the network, use new to obtain same type of variable
        # weights are (n_layers*n_directions, batch_size, hidden_dim)
        hidd = weight.new(self.n_layers*2, batch_size, self.hidden_dim)
        cell = weight.new(self.n_layers*2, batch_size, self.hidden_dim)
                
        nn.init.xavier_normal_(hidd) # in-place xavier init        
        nn.init.xavier_normal_(cell) 
        
        if (self.train_on_gpu):
            return ( hidd.cuda(), cell.cuda() )
        else:
            return ( hidd, cell )        
        
class SimpleLSTMDecoderLayer(nn.Module):        
    def __init__(self, vocab_size, embedding_dim, encoder_output_dim, hidden_dim, n_layers, drop_prob=0.3):        
        super(SimpleLSTMDecoderLayer, self).__init__()

        self.vocab_size = vocab_size
        self.input_dim = embedding_dim + encoder_output_dim
        self.hidden_dim = hidden_dim
        self.output_dim = vocab_size
        self.n_layers = n_layers
        self.train_on_gpu=torch.cuda.is_available()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # I should not have 2 embedding layers. maybe move in top class and work with embeddings only
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.softmax_projection = nn.Linear(hidden_dim, vocab_size)
    
    def forward_step(self, prev_y, prev_decoder_hidden, encoder_output):      
        # prev_embedding is a batch_size * 1 containing 1 word index (previous)
        prev_y = self.embedding(prev_y)
        # prev_embedding is a batch_size * 1 * embedding_dim containing 1 word embedding (previous)
        
        # encoder_output is batch_size x enc_hidden_dim
        #print(prev_y.size())
        #print(encoder_output.size())
        
        # update rnn hidden state
        input = torch.cat([prev_y, encoder_output.unsqueeze(1)], dim=2)
        output, decoder_hidden = self.lstm(input, prev_decoder_hidden)
        
        #word_softmax_projection = torch.cat([prev_y, output, context], dim=2)         ???
        word_softmax_projection = self.softmax_projection(output)

        return output, decoder_hidden, word_softmax_projection

    def init_hidden(self, batch_size): 
        weight = next(self.parameters()).data # get any parameter of the network, use new to obtain same type of variable
        h_a = weight.new(self.n_layers, batch_size, self.hidden_dim)
        h_b = weight.new(self.n_layers, batch_size, self.hidden_dim)
                
        nn.init.xavier_normal_(h_a) # in-place xavier init        
        nn.init.xavier_normal_(h_b) 
        
        if (self.train_on_gpu):
            return ( h_a.cuda(), h_b.cuda() )
        else:
            return ( h_a, h_b )   

class AttentionLayer(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super(AttentionLayer, self).__init__()

        # attention weights
        self.attention_w1 = nn.Linear(enc_hidden_dim, enc_hidden_dim) #self.model.add_parameters((enc_state_size, enc_state_size))
        self.attention_w2 = nn.Linear(dec_hidden_dim, enc_hidden_dim) #self.model.add_parameters((enc_state_size, dec_state_size))
        self.attention_v = nn.Linear(enc_hidden_dim, 1) #self.model.add_parameters((1, enc_state_size))
        self.softmax = nn.Softmax(dim=1)
        
        self.enc_hidden_dim = enc_hidden_dim
    """
    def _attend(self, input_vectors, state):
        w1 = dy.parameter(self.attention_w1)
        w2 = dy.parameter(self.attention_w2)
        v = dy.parameter(self.attention_v)
        attention_weights = []

        w2dt = w2 * state.h()[-1]
        for input_vector in input_vectors:
            attention_weight = v * dy.tanh(w1 * input_vector + w2dt)
            attention_weights.append(attention_weight)
        attention_weights = dy.softmax(dy.concatenate(attention_weights))

        output_vectors = dy.esum(
            [vector * attention_weight for vector, attention_weight in zip(input_vectors, attention_weights)])
        return output_vectors
    """
    
    def forward(self, encoder_outputs, decoder_output): 
        # encoder_outputs is batch_size * max_seq_len_x * encoder_hidden
        # decoder_output at prev timestep: (batch, seq_len, num_directions * hidden_size) (because batch_first = True)
        
        # seq_len = 1 because there's just one timestep, num_directions = 1 because the decoder is unidirectional
        # so, decoder_output is (batch_size, 1, decoder_hidden_size)        
        # we want to return context is an encoder_hidden_size vector of shape (batch_size, encoder_hidden_size)
        #print(encoder_outputs.size())
        #print(decoder_output.size())
       
        # project fixed decoder_output
        w2_decoder_output = self.attention_w2( decoder_output )                
        #print(w2_decoder_output.size())
        
        # now, we calculate, for all enc_outputs, their transformation through the w1 linear layer
        w1_transformed_encoder_outputs = self.attention_w1(encoder_outputs)
        #print(w1_transformed_encoder_outputs.size())
        
        # add w2_decoder_output to each of the max_seq_len_x elements in w1_transformed_encoder_outputs
        
        w1_w2_sum = w1_transformed_encoder_outputs + w2_decoder_output
        #print(w1_w2_sum.size())
        #print("{} + {} = {}".format(w1_transformed_encoder_outputs[0][0][0], w2_decoder_output[0][0][0], w1_w2_sum[0][0][0]))
        
        # tanh everything
        w1_w2_sum_tanh = w1_w2_sum.tanh()
        
        # transform each of the encoder states to a single value        
        attention_weights = self.attention_v(w1_w2_sum_tanh)
        #print(attention_weights.size()) # size is batch_size x max_seq_len_x x 1
        
        softmax_attention_weights = self.softmax(attention_weights.squeeze(2)) # squeeze last 1 dimension so we softmax on each of the max_seq_len_x elements (sum to 1)        
        #print(softmax_attention_weights.size()) # size is batch_size x max_seq_len_x
    
        # now, the context is the element-wise sum of each of the encoder_outputs (transformed so far)
        # first, we multiply each encoder_output with its attention value        
        # encoder_outputs is batch_size * max_seq_len_x * encoder_hidden
        # softmax_attention_weights is batch_size x max_seq_len_x so we unsqueeze(2) the last dim to make it batch_size x max_seq_len_x x 1, and then copy the last value encoder_hidden_dim times with expand()
        weighted_encoder_outputs = encoder_outputs * softmax_attention_weights.unsqueeze(2).expand(-1,-1,self.enc_hidden_dim)
        # the weighted_encoder_outputs is batch_size * max_seq_len_x * encoder_hidden , just like encoder_outputs
        
        # element wise sum on each of the max_seq_len_x elements to obtain final context of batch_size * encoder_hidden
        context = weighted_encoder_outputs.sum(dim=1) 
        
        return context
        
       
            
if __name__ == '__main__':
    
    net = Attention(512, 256)

    encoder_outputs = torch.Tensor(64,399,512).float().random_(0,1)
    decoder_hidden = torch.Tensor(64,1,256).float().random_(0,1)
    
    out = net(encoder_outputs, decoder_hidden)            
    
    print(out.size())
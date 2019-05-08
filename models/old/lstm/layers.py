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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
     
    def forward(self, x, hidden):        
        batch_size = x.size(0)
        #64 x 399 
        x = self.embedding(x)
        #64 x 399 x embedding_dim
        
        # embeddings and lstm_out        
        lstm_out, hidden = self.lstm(x, hidden)
        # lstm_out is 64 x 399 * hidden_dim

        return lstm_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (self.train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden
        
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
        
        #print(prev_y.size())
        #print(encoder_output.size())
        
        # update rnn hidden state
        input = torch.cat([prev_y, encoder_output], dim=2)
        output, decoder_hidden = self.lstm(input, prev_decoder_hidden)
        
        #word_softmax_projection = torch.cat([prev_y, output, context], dim=2)         ???
        word_softmax_projection = self.softmax_projection(output)

        return output, decoder_hidden, word_softmax_projection

    
    def forward(self, input, y):        
        batch_size = input.size(0)
        max_seq_len_y = y.size(1)
        
        decoder_hidden = self.init_hidden(batch_size)
        
        decoder_input = torch.zeros(batch_size, 1)
        decoder_input[:,0] = 0 # BOS = 0
        
        # prepare decoder output as batch_size * max_seq_len_x * vocab_size
        decoder_output = torch.zeros(batch_size, max_seq_len_y, self.vocab_size)#.to(self.device)
                        
        #for i in range(max_seq_len_y):
            
        
        #for i in 
        
        lstm_out, hidden = self.lstm(input, hidden)
    
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
       
        # reshape to be batch_size first
        out = out.view(batch_size, -1)
        out = out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (self.train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden    
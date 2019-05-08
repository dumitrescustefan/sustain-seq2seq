import torch
import torch.nn as nn
import math 
import numpy as np
import random

from pprint import pprint

class SimpleLSTMEncoderLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.3):        
        super(SimpleLSTMEncoderLayer, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.train_on_gpu=torch.cuda.is_available()
        self.vocab_size = vocab_size
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop_prob)
     
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
        
        lstm_out = self.dropout(lstm_out)        
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
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim) 
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

class DroppedLSTMDecoderLayer(nn.Module):        
    def __init__(self, vocab_size, embedding_dim, encoder_output_dim, hidden_dim, n_layers, drop_prob=0.3, input_word_drop=0.5):        
        super(DroppedLSTMDecoderLayer, self).__init__()

        self.vocab_size = vocab_size
        self.input_dim = embedding_dim + encoder_output_dim
        self.hidden_dim = hidden_dim
        self.output_dim = vocab_size
        self.n_layers = n_layers
        self.train_on_gpu = torch.cuda.is_available()
        self.input_word_drop = input_word_drop
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim) 
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.softmax_projection = nn.Linear(hidden_dim, vocab_size)
    
    def forward_step(self, prev_y, prev_decoder_hidden, encoder_output, force_keep_prev_y=False):      
        # prev_embedding is a batch_size * 1 containing 1 word index (previous)
        if random.random()<=self.input_word_drop and force_keep_prev_y == False:
            prev_y = torch.ones(prev_y.size(0),1, dtype = torch.long, device = self.device) # <UNK>                        
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

            
class VAE(nn.Module):            
    def __init__(self, input_size, latent_size):
        super(VAE, self).__init__()
        
        self.latent_size = latent_size
        self.input_size = input_size
        self.mu_layer = nn.Linear(self.input_size, self.latent_size)
        self.logvar_layer = nn.Linear(self.input_size, self.latent_size)
         
    def _reparametrize(self, mu, logvar):
        std_dev = torch.exp(0.5*logvar)
        eps = torch.randn_like(std_dev)
        return eps.mul(std_dev).add_(mu) #z = z * std + mu
    
    def forward(self, input):        
        mu = self.mu_layer(input)
        logvar = self.logvar_layer(input)        
        z = self._reparametrize(mu, logvar)
        return z, mu, logvar
    
    def default_loss(self, x, recon_x, mu, logvar): # we don't actually use this, it's in the training code
        BCE = nn.functional.cross_entropy(recon_x, x)
        #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')            
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    def kl_anneal_function(self, step, k=0.0025, x0=2500, anneal_function="linear"):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)        
            
class AttentionLayer(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super(AttentionLayer, self).__init__()

        # attention weights
        self.attention_w1 = nn.Linear(enc_hidden_dim, enc_hidden_dim) #self.model.add_parameters((enc_state_size, enc_state_size))
        self.attention_w2 = nn.Linear(dec_hidden_dim, enc_hidden_dim) #self.model.add_parameters((enc_state_size, dec_state_size))
        self.attention_v = nn.Linear(enc_hidden_dim, 1) #self.model.add_parameters((1, enc_state_size))
        self.softmax = nn.Softmax(dim=1)
        
        self.enc_hidden_dim = enc_hidden_dim
        
        self.should_print = False
        self.att_mat = []

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
        
        if self.should_print:
            to_cpu = softmax_attention_weights.cpu()
            row = to_cpu[0].data.numpy()            
            self.att_mat.append(row)
            #print (len(self.att_mat))
    
        # now, the context is the element-wise sum of each of the encoder_outputs (transformed so far)
        # first, we multiply each encoder_output with its attention value        
        # encoder_outputs is batch_size * max_seq_len_x * encoder_hidden
        # softmax_attention_weights is batch_size x max_seq_len_x so we unsqueeze(2) the last dim to make it batch_size x max_seq_len_x x 1, and then copy the last value encoder_hidden_dim times with expand()
        weighted_encoder_outputs = encoder_outputs * softmax_attention_weights.unsqueeze(2).expand(-1,-1,self.enc_hidden_dim)
        # the weighted_encoder_outputs is batch_size * max_seq_len_x * encoder_hidden , just like encoder_outputs
        
        # element wise sum on each of the max_seq_len_x elements to obtain final context of batch_size * encoder_hidden
        context = weighted_encoder_outputs.sum(dim=1) 
        
        return context
        
class Beam():
    def __init__(self, alpha = 0.7):#, beam_size):
        #self.beam_size = beam_size
        self.past_decoder_hidden = None
        self.current_decoder_hidden = None
        self.score = 0.
        self.sequence = []        
        self.alpha = alpha
    
    def normalized_score(self):
        return self.score / math.pow(len(self.sequence),self.alpha)
        
    def ended(self):
        return False if self.sequence[-1] != 0 else True

        
class not_used_AbsolutePositionEmbeddings(nn.Module):
    def __init__(self, max_seq_len, embedding_dim):    
        super(AbsolutePositionEmbeddings,self).__init__() # max_seq_len + 1 because zero index has all zeroes       
        self.embedding = nn.Embedding(vocab_size, embedding_dim) 
        
        # init static position embedding
        position_enc = np.array([ [pos / np.power(10000, 2 * (j // 2) / embedding_dim) for j in range(embedding_dim)] if pos != 0 else np.zeros(embedding_dim) for pos in range(max_seq_len+1)]) 
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) 
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) 
        
        self.max_seq_len = max_seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.absolute_position_encoder = torch.nn.Embedding(max_seq_len+1, embedding_dim, padding_idx=0)
        self.absolute_position_encoder.weight.data = torch.FloatTensor(position_enc)
        
    def forward(self, sequence, sequence_lenghts):
        batch_size = sequence.size(0) # sequence should be (bs,seq_len,emb_dim), sequence_lenghts is an array like [3,4,k.. ] of len batch_size with the last non-zero element
        lengths = [] 
        for i in range(batch_size):
            li = np.arange(1, sequence_lenghts[i]+1, dtype=long)
            if sequence_lenghts[i]<self.max_seq_len: # pad with zeroes
                li += np.zeros(self.max_seq_len - sequence_lenghts[i], dtype=long)                
            lengths.append(li)
        lengths = torch.tensor(lengths, device = self.device)        
        # lengths is an array of (bs, seq_len) with each row as [1,2,3, k, 0,0..0] of len max_seq_len        
        return sequence + self.absolute_position_encoder(lengths) # return is same size as input (bs,seq_len,emb_dim) 

class SelfAttention(nn.Module): # adapted from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob = 0.2):#, hidden_dropout_prob = 0.2):
        super(SelfAttention, self).__init__()        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        
        #self.dense = nn.Linear(hidden_size, hidden_size)
        self.layernorm = LayerNorm(hidden_size, eps=1e-12)
        #self.output_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)
        print(mixed_key_layer.size())

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        print(query_layer.size())
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        print(attention_scores.size())
        
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)        
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        print(context_layer.size())
        print( context_layer.size()[:-2])
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        print( new_context_layer_shape )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
        
        #hidden_states = self.dense(hidden_states)
        #hidden_states = self.dropout(hidden_states)
        context_layer = self.layernorm(context_layer + input_tensor) # skip connection
        return context_layer

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):        
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfOutput(nn.Module):
    def __init__(self, config):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob = 0.2):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob = 0.2)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output        
        

        
if __name__ == '__main__':
    
    """net = Attention(512, 256)

    encoder_outputs = torch.Tensor(64,399,512).float().random_(0,1)
    decoder_hidden = torch.Tensor(64,1,256).float().random_(0,1)
    
    out = net(encoder_outputs, decoder_hidden)            
    
    print(out.size())
    """
    
    net = SelfAttention(512, 8) # hidden, att_heads

    input_tensor = torch.Tensor(4, 10, 512).float().random_(0,1)
    attention_mask = torch.ones(4, 10)
    #attention_mask = torch.ones_like(input_ids)
        
    # We create a 3D attention mask from a 2D tensor mask.
    # Sizes are [batch_size, 1, 1, to_seq_length]
    # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
    # this attention mask is more simple than the triangular masking of causal attention
    # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    #extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    
    out = net(input_tensor, extended_attention_mask)            
    
    

    print(out.size())
    
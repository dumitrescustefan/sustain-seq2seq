import torch
import torch.nn as nn
import math
import numpy as np
          
class AdditiveAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super(AdditiveAttention, self).__init__()

        # attention weights
        self.attention_w1 = nn.Linear(enc_hidden_dim, enc_hidden_dim) #self.model.add_parameters((enc_state_size, enc_state_size))
        self.attention_w2 = nn.Linear(dec_hidden_dim, enc_hidden_dim) #self.model.add_parameters((enc_state_size, dec_state_size))
        self.attention_v = nn.Linear(enc_hidden_dim, 1) #self.model.add_parameters((1, enc_state_size))
        self.softmax = nn.Softmax(dim=1)
        
        self.enc_hidden_dim = enc_hidden_dim
        
        self.should_print = False
        self.att_mat = []

    
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


#testing 
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
    
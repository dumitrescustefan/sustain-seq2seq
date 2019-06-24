import os, sys, math
sys.path.insert(0, '../../..')

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):        
    """
        Input: q, k, v are [batch_size, seq_len, d_model] 
        Output: 
    """
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads        
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_head)
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_head)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_head)
        
        # transpose to get dimensions bs * h * sl * d_model       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        # calculate attention using function we will define next
        scores = self._attention(q, k, v, self.d_head, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output
        
    def _attention(self, q, k, v, d_k, mask=None, dropout=None):    
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            scores = dropout(scores)
            
        output = torch.matmul(scores, v)
        return output    
        
if __name__ == "__main__":
    import numpy as np
    
    
    # debug stuff:    
    q = torch.tensor([ [ [2.,2.,2.] ] ])
    K = torch.tensor([ [ [1.,1.,1.] , [5.,5.,5.] ] ])
    #result would be ([ [ [2.,2.,2.] , [2.5,2.5,2.5] ] ])
    
    print(K.size())
    print(q)
    print(q.size())
    #qt = q.expand(1,3,3)#q.transpose(1,2)
    qt = q.transpose(1,2)
    print(qt)
    print(qt.size())    
    
    print()
    r = torch.bmm(K,qt)
    print(r)
    print(r.size())
    print()
    #print(e1.size())
    #print(v1.size())
    #qq = e1*v1
    #print(qq)
    
    
    
    # prep inputs
    batch_size = 2
    seq_len = 10
    enc_size = 4
    dec_layers = 5
    dec_size = 3
    
    encoder_outputs = torch.tensor(np.random.rand(batch_size, seq_len, enc_size), dtype=torch.float)
    decoder_hidden_state = torch.tensor(np.random.rand(dec_layers*1, batch_size, dec_size), dtype=torch.float) # 1 for unidirectional
    
    # prep layer
    device = torch.device("cpu")
    #type = "additive"    
    type = "general"    
    att = Attention(enc_size, dec_size, device, type)
    
    # run
    context, attention_weights = att(encoder_outputs, decoder_hidden_state)
    print("Output is:")
    print(context)
    print("Attention weights size:" + str(attention_weights.size()))
    
    # debug stuff:    
    #e1 = torch.tensor([[[2],[0.5]]])
    #v1 = torch.tensor([ [ [1.,1.,1.] , [5.,5.,5.] ] ])
    #result would be ([ [ [2.,2.,2.] , [2.5,2.5,2.5] ] ])
    #print(e1.size())
    #print(v1.size())
    #qq = e1*v1
    #print(qq)

            
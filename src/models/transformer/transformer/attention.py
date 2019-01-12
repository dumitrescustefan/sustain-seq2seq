import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # print(q.size()) torch.Size([64, 399, 512]), all the same        
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        # w_qs is a Linear (in = 512, out = 8 heads * 64 key/query/value size)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) # first pass through linear, then reshape result -> .view(64, 399, 8, 64)
        # q is now (64, 399, 8, 64)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        #print(q.size())
        #print(q.permute(2, 0, 1, 3).size())
        # torch.Size([64, 399, 8, 64])
        # torch.Size([8, 64, 399, 64])
                
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk -> 
        # torch.Size([8, 64, 399, 64]) reshape into ((8*64), 399, 64)
        #print(q.size())
        #torch.Size([512, 399, 64])
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        # print(mask.size()) torch.Size([64, 399, 399])
        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        # print(mask.size()) torch.Size([512, 399, 399])
        # so, q,k,v shapes are [512, 399, 64]
        output, attn = self.attention(q, k, v, mask=mask)
        # output is torch.Size([512, 399, 64])
        # attn is [512, 399, 399]
        output = output.view(n_head, sz_b, len_q, d_v) # 8, 64, 399, 64
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        
        # print(output.size()) torch.Size([64, 399, 512])
        
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn
        
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        
        attn = torch.bmm(q, k.transpose(1, 2)) # Performs a batch matrix-matrix product of matrices stored in batch1 and batch2.
        attn = attn / self.temperature
        # print(attn.size()) [512, 399, 399]

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # v is [512, 399, 64]
        # so bmm is [512, 399, 399] * [512, 399, 64] => [512, 399, 64]
        output = torch.bmm(attn, v)
        # print(output.size()) torch.Size([512, 399, 64])
        return output, attn     
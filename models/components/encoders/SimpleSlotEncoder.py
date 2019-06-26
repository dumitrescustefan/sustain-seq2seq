import os, sys
sys.path.insert(0, '../../..')

import torch
import torch.nn as nn

class SimpleSlotEncoder(nn.Module):
    def __init__(self, emb_dim, slot_sizes, dropout, device):
        super(SimpleSlotEncoder, self).__init__()

        self.embeddings = []
        for i in range(len(slot_sizes)):
            self.embeddings.append(nn.Embedding(slot_sizes[i], emb_dim))
            
        for i in range(len(slot_sizes)):
            self.embeddings[i].cuda()
        self.dropout = nn.Dropout(dropout)
        
        self.device = device
        self.emb_dim = emb_dim
        self.to(device)

    def forward(self, input):
        """
        Input: fixed string of ints [batch_size, seq_len], where each int is the index of the category in that particular slot

        Returns:
            Embeddings like [batch_size, seq_len, emb_dim]        
        """

        # Creates the embeddings and adds dropout. [batch_size, seq_len] -> [batch_size, seq_len, emb_dim].
        batch_size = input.size(0)
        seq_len = input.size(1)
        output = torch.zeros(batch_size, seq_len, self.emb_dim, device = self.device)                
        
        for i in range(seq_len):            
            output[:,i:i+1,:] = self.dropout(self.embeddings[i](input[:,i:i+1]))
             
        return output
        
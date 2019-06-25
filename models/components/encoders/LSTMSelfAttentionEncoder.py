import os, sys
sys.path.insert(0, '../../..')

import torch
import torch.nn as nn
from models.components.attention.MultiHeadAttention import MultiHeadAttention

class LSTMSelfAttentionEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, lstm_dropout, dropout, device):
        """
        Creates an Encoder model.

        Args:
            vocab_size (int): Number of classes/ Vocabulary size.
            emb_dim (int): Embeddings dimension.
            hidden_dim (int): LSTM hidden layer dimension.
            num_layers (int): Number of LSTM layers.
            lstm_dropout (float): LSTM dropout.
            dropout (float): Embeddings dropout.
            device : The device to run the model on.
        """
        assert hidden_dim % 2 == 0, "LSTMSelfAttentionEncoder hidden_dim should be even as the LSTM is bidirectional."
        super(LSTMSelfAttentionEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)        
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, int(hidden_dim/2), num_layers, dropout=lstm_dropout, bidirectional=True, batch_first=True)
        
        num_heads = 8 # TODO parametrize this
        assert hidden_dim % num_heads == 0, "LSTMSelfAttentionEncoder hidden_dim ({}) should be a multiple of num_heads ({}).".format(hidden_dim, num_heads)        
        
        self.self_attention = MultiHeadAttention(d_model=emb_dim, num_heads=num_heads, dropout=dropout)
        
        self.to(device)

    def forward(self, input):
        """
        Args:
            input (tensor): The input of the encoder. It must be a 2-D tensor of integers. 
                Shape: [batch_size, seq_len_enc].

        Returns:
            A tuple containing the output and the states of the last LSTM layer. The states of the LSTM layer is also a
            tuple that contains the hidden and the cell state, respectively . 
                Output shape:            [batch_size, seq_len_enc, hidden_dim * 2]
                Hidden/cell state shape: [num_layers*2, batch_size, hidden_dim]
        """

        # Creates the embeddings and adds dropout. [batch_size, seq_len] -> [batch_size, seq_len, emb_dim].
        embeddings = self.dropout(self.embedding(input))

        self_attn = self.self_attention(q=embeddings, k=embeddings, v=embeddings, mask=None)
        
        # Computes the output and the two states of the lstm layer. See function returns docs for details.
        lstm_output, states = self.lstm(self_attn)

        # multihead attention
        #self_attn = self.self_attention(q=lstm_output, k=lstm_output, v=lstm_output, mask=None)
        
        #output = torch.cat((lstm_output, self_attn), dim = 2)
        
        return lstm_output, states
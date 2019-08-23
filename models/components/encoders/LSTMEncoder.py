import os, sys
sys.path.insert(0, '../../..')

import torch
import torch.nn as nn

class Encoder(nn.Module):
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
        assert hidden_dim % 2 == 0, "Encoder hidden_dim should be even as the LSTM is bidirectional."
        super().__init__()
        
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers        

        self.embedding = nn.Embedding(vocab_size, emb_dim)        
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, int(hidden_dim/2), num_layers, dropout=lstm_dropout, bidirectional=True, batch_first=True)

        self.to(device)

    def forward(self, input_tuple):
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
        
        X, X_lengths = input_tuple[0], input_tuple[1]
        
        # Creates the embeddings and adds dropout. [batch_size, seq_len] -> [batch_size, seq_len, emb_dim].
        embeddings = self.dropout(self.embedding(X))
        
        # pack padded sequences
        pack_padded_lstm_input = torch.nn.utils.rnn.pack_padded_sequence(embeddings, X_lengths, batch_first=True)

        # now run through LSTM
        pack_padded_lstm_output, states = self.lstm(pack_padded_lstm_input)
        
        # undo the packing operation
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_padded_lstm_output, batch_first=True)        
        
        return {'output':output, 'states':states}
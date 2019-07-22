import sys

sys.path.insert(0, '../../..')

import torch.nn as nn


class LSTMDecoder(nn.Module):
    def __init__(self, emb_dim, input_size, hidden_dim, num_layers, n_class, lstm_dropout, dropout, device):
        """
        Creates an Decoder with attention.

        Args:
            emb_dim (int): Embeddings dimension.
            input_size (int): Size of input (E.g. size of encoder's output / hidden_dim*2 if bidirectional lstm, etc.)
            hidden_dim (int): LSTM hidden layers dimension.
            num_layers (int): Number of LSTM layers.
            n_class (int): Number of classes/ Vocabulary size.
            lstm_dropout (float): LSTM dropout.
            dropout (float): Embeddings dropout 
            device : The device to run the model on.
        """

        super(LSTMDecoder, self).__init__()

        self.embedding = nn.Embedding(n_class, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim + input_size, hidden_dim, num_layers, dropout=lstm_dropout, batch_first=True)
        self.output_linear = nn.Linear(hidden_dim, n_class)

        self.device = device
        self.to(device)

    def forward(self, x_tuple, y_tuple, enc_output, dec_states, teacher_forcing_ratio):
        """
        Args:
             input (tensor): The input of the decoder.
                Shape: [batch_size, seq_len_dec].
             enc_output (tensor): The output of the encoder last LSTM layer.
                Shape: [batch_size seq_len_enc, hidden_dim * 2].
             dec_states (tuple of tensors): The hidden and the cell states of the encoder last LSTM layer after being transformed by the linear layer.
                State shape: [num_layers, batch_size, hidden_dim].

        Returns:
            The output of the decoder, a tensor that contains a sequence of tokens with the dimension equal to
            vocabulary size. Shape: [batch_size, seq_len_dec, n_class].
        """
        pass
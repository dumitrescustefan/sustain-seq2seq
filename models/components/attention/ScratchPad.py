import os, sys
sys.path.insert(0, '../../..')

import torch
import torch.nn as nn

class ScratchPad(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, lstm_dropout, dropout, device):
        """
        https://arxiv.org/pdf/1906.05275v2.pdf
        bupL https://arxiv.org/abs/1808.10792
        """
        assert hidden_dim % 2 == 0, "LSTMEncoder hidden_dim should be even as the LSTM is bidirectional."
        super(ScratchPad, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)        
        self.dropout = nn.Dropout(dropout)
        self.mlp_alpha = nn.Linear()
        self.mlp_u = nn.Linear()

        self.to(device)

    def forward(self, encoder_output, decoder_state, context):
        """
        Args:
            encoder_output (tensor): The encoder's output
                Shape:      [batch_size, seq_len_enc, enc_size]
            decoder_state:  [num_layers, batch_size, hidden_dim]
            context:        [batch_size, encoder_size]
        Returns:
            Rewritten encoder outputs:
                Shape:      [batch_size, seq_len_enc, enc_size]
        """
        
        # first, compute alpha_i as sigmoid(mlp(state, context, ))

    
        return output, states
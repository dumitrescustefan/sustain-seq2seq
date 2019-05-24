import sys
sys.path.insert(0, '../../..')

import torch
import torch.nn as nn

from models.components.attention.CoverageAdditiveAttention import CoverageAdditiveAttention
from models.components.decoders.LSTMDecoder import LSTMDecoder

import numpy as np


class LSTMDecoderWithAdditiveAttentionCoverage(LSTMDecoder):
    def __init__(self, emb_dim, input_size, hidden_dim, num_layers, n_class, lstm_dropout, dropout, device, fertiliy=5):
        """
        Creates a Decoder with attention and coverage.

        Args :
            dropout (float): The dropout in the attention layer.

            See LSTMDecoder for further args info
        """

        super(LSTMDecoderWithAdditiveAttentionCoverage, self).__init__(emb_dim, input_size, hidden_dim, num_layers, n_class, lstm_dropout, device)

        self.coverage_attention = CoverageAdditiveAttention(encoder_input_size=input_size, decoder_hidden_size=hidden_dim, dropout=dropout, device=device)
        self.lstm = nn.LSTM(emb_dim + input_size + hidden_dim, hidden_dim, num_layers, dropout=lstm_dropout, batch_first=True)

        self.to(device)

    def forward(self, input, enc_output, dec_states, teacher_forcing_ratio):
        """
            See LSTMDecoder for further info.
        """

        batch_size = input.shape[0]
        seq_len_dec = input.shape[1]
        seq_len_enc = enc_output.shape[1]

        dec_states = (dec_states[0].contiguous(), dec_states[1].contiguous())
        output = torch.Tensor().to(self.device)

        # Initializes the coverage with a normal distribution of mean 0 and sigma 0.01
        n_dist = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([0.01]))
        coverage = n_dist.sample([batch_size, seq_len_enc]).to(self.device)

        # Loop over the rest of tokens in the input seq_len_dec.
        for i in range(0, seq_len_dec - 1):
            # Calculate the context vector at step i.
            context_vector, coverage = self.coverage_attention(coverage, dec_states[0], enc_output)

            if np.random.uniform(0, 1) < teacher_forcing_ratio or i is 0:
                # Concatenates the i-th embedding of the input with the corresponding  context vector over the second
                # dimensions. Transforms the 2-D tensor to 3-D sequence tensor with length 1. [batch_size, emb_dim] +
                # [batch_size, hidden_dim * num_layers] -> [batch_size, 1, emb_dim + hidden_dim * num_layers].
                lstm_input = torch.cat((self.embedding(input[:, i]), context_vector), dim=1).reshape(batch_size, 1, -1)
            else:
                # Calculates the embeddings of the previous output. Counts the argmax over the last third dimension and
                # then squeezes the second dimension, the sequence length. [batch_size, emb_dim].
                prev_output_embeddings = self.embedding(torch.squeeze(torch.argmax(lin_output, dim=2), dim=1))

                # Concatenates the (i-1)-th embedding of the previous output with the corresponding  context vector over the second
                # dimensions. Transforms the 2-D tensor to 3-D sequence tensor with length 1. [batch_size, emb_dim] +
                # [batch_size, hidden_dim * num_layers] -> [batch_size, 1, emb_dim + hidden_dim * num_layers].
                lstm_input = torch.cat((prev_output_embeddings, context_vector), dim=1).reshape(batch_size, 1, -1)

            # Calculates the i-th decoder output and state. We initialize the decoder state with (i-1)-th state.
            # [batch_size, 1, hidden_dim], [num_layers, batch_size, hidden_dim].
            dec_output, dec_states = self.lstm(lstm_input, dec_states)

            # Maps the decoder output to the decoder vocab size space. [batch_size, 1, hidden_dim] -> [batch_size, 1,
            # n_class].
            lin_output = self.output_linear(dec_output)

            # Adds the current output to the final output. [batch_size, i-1, n_class] -> [batch_size, i, n_class].
            output = torch.cat((output, lin_output), dim=1)

        return output

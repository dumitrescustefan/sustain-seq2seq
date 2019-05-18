import torch.nn as nn
from models.components.attention.AdditiveAttention import AdditiveAttention
import torch
from models.components.attention.Attention import Attention


class CoverageAdditiveAttention(Attention):
    def __init__(self, encoder_input_size, decoder_hidden_size, dropout, device, N=2):
        """
            Args:
                N (int): the maximum number of outputs, an input can create.
        """

        super(CoverageAdditiveAttention, self).__init__()

        self.N = N

        self.additive_attn = AdditiveAttention(encoder_input_size, decoder_hidden_size * 2, dropout, device)

        self.coverage = nn.Linear(1, decoder_hidden_size)

        self.fertility = nn.Linear(decoder_hidden_size, 1)

        self.to(device)

    def forward(self, coverage, state_h, enc_output):
        """
            Args:
                coverage (tensor): the coverage vector. Shape: [batch_size, seq_len_enc, 1]

            Returns:
                A tuple containing the context vector and the coverage vector. Shape: [batch_size, dec_hidden_size],
                [batchs_size, seq_len_enc, 1].

        """
        # Creates a new input vector for the attention by concatenating coverage and the output of the encoder.
        # [batch_size, seq_len_enc, encoder_input_size] + [batch_size, seq_len_enc, decoder_hidden_size] ->
        # [batch_size, seq_len_enc, encoder_input_size + decoder_hidden_size].
        attn_input = torch.cat((enc_output, self.coverage(coverage)), dim=2)

        # Create the context vector and the attention weights.
        context_vector, attn_weights = self.additive_attn(state_h, attn_input)

        # Modify the shape of the hidden state.
        batch_size = enc_output.shape[0]
        seq_len = enc_output.shape[1]
        state_h = self.calculate_new_state_h(state_h, batch_size, seq_len)

        # Calculate the fertility of each input. It is calculated as 1 divided by the percentage of N(the maximum number of outputs
        # an input can produce). [batch_size, seq_len_enc, 1].
        fertility = 1 / (self.N * torch.sigmoid(self.fertility(state_h)))

        # Calculates the new coverage. Multiply the attention weights by the fertility and add the previous coverage vector.
        # [batch_size, seq_len_enc, 1].
        coverage = torch.add(fertility * attn_weights, coverage)

        return context_vector, coverage
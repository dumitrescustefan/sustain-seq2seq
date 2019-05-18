import torch.nn as nn


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def calculate_new_state_h(self, state_h, batch_size, seq_len):
        """
        Reshapes the hidden state to desired shape: [num_layers, batch_size, decoder_hidden_size] -> batch_size, seq_len_enc,
        decoder_hidden_state].

        Args:
            state_h (tensor): Previous hidden state of the decoder.
            batch_size (int): The size of the batch.
            seq_len (int): The length of the encoder sequence.

        Returns:
            The reshaped hidden state.
        """

        # in case the decoder has more than 1 layer, take only the last one
        if state_h.shape[0] > 1:
            state_h = state_h[state_h.shape[0]-1:state_h.shape[0],:,:]

        # [dec_num_layers * 1, batch_size, decoder_hidden_size] -> [batch_size, dec_num_layers * 1, decoder_hidden_size]
        state_h = state_h.permute(1, 0, 2)

        # [batch_size, dec_num_layers * 1, decoder_hidden_size] -> [batch_size, 1, dec_num_layers * 1 * decoder_hidden_size]
        state_h = state_h.reshape(batch_size, 1, -1)

        # [batch_size, 1, dec_num_layers * 1 * decoder_hidden_size] -> [batch_size, seq_len, dec_num_layers * 1 * decoder_hidden_size]
        state_h = state_h.expand(-1, seq_len, -1)

        return state_h
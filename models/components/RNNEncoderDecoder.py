import os, sys

sys.path.insert(0, '../..')

import torch
import torch.nn as nn

class RNNEncoderDecoder(nn.Module):
    def __init__(self,
                 # encoder params
                 rnn_encoder_object, enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers, enc_lstm_dropout, enc_dropout, 
                 # decoder params
                 rnn_decoder_object, dec_input_dim, dec_emb_dim, dec_hidden_dim, dec_num_layers, dec_vocab_size, dec_lstm_dropout, dec_dropout, dec_attention_type, dec_transfer_hidden=True):

        super(RNNEncoderDecoder, self).__init__()

        if torch.cuda.is_available():
            print('Running on GPU.')
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            print('Running on CPU.')
            self.cuda = False
            self.device = torch.device('cpu')

        self.enc_vocab_size = enc_vocab_size
        self.enc_emb_dim = enc_emb_dim
        self.enc_hidden_dim = enc_hidden_dim
        
        self.dec_input_dim = dec_input_dim
        self.dec_emb_dim = dec_emb_dim        
        self.dec_hidden_dim = dec_hidden_dim
        self.dec_num_layers = dec_num_layers
        self.dec_transfer_hidden = dec_transfer_hidden
        self.dec_vocab_size = dec_vocab_size
        self.dec_attention_type = dec_attention_type

        if dec_transfer_hidden == True:
            assert enc_num_layers == dec_num_layers, "For transferring the last hidden state from encoder to decoder, both must have the same number of layers."

        self.encoder = rnn_encoder_object(vocab_size=enc_vocab_size, emb_dim=enc_emb_dim, hidden_dim=enc_hidden_dim,
                            num_layers=enc_num_layers, lstm_dropout=enc_lstm_dropout, dropout=enc_dropout, device=self.device)
        
        self.decoder = rnn_decoder_object(emb_dim=dec_emb_dim, input_size=enc_hidden_dim, hidden_dim=dec_hidden_dim, num_layers=dec_num_layers,
                            n_class=dec_vocab_size, lstm_dropout=dec_lstm_dropout, dropout=dec_dropout, attention_type=dec_attention_type, device=self.device)

        # Transform h from encoder's [num_layers * 2, batch_size, enc_hidden_dim/2] to decoder's [num_layers * 1, batch_size, dec_hidden_dim], same for c; batch_size = 1 (last timestep only)
        self.h_state_linear = nn.Linear(int(enc_hidden_dim * enc_num_layers/1), dec_hidden_dim * dec_num_layers * 1)
        self.c_state_linear = nn.Linear(int(enc_hidden_dim * enc_num_layers/1), dec_hidden_dim * dec_num_layers * 1)

        self.to(self.device)

    def forward(self, x, y, teacher_forcing_ratio=0.):
        """
        Args:
            x (tensor): The input of the decoder. Shape: [batch_size, seq_len_enc].
            y (tensor): The input of the decoder. Shape: [batch_size, seq_len_dec].

        Returns:
            The output of the Encoder-Decoder with attention. Shape: [batch_size, seq_len_dec, n_class].
        """
        batch_size = x.shape[0]

        # Calculates the output of the encoder
        enc_output, enc_states = self.encoder.forward(x)
        # enc_states is a tuple of size ( h=[enc_num_layers*2, batch_size, enc_hidden_dim/2], c=[same-as-h] )

        if self.dec_transfer_hidden == True:
            dec_states = self.transfer_hidden_from_encoder_to_decoder(enc_states)
        else:
            pass  # dec_state = (,) # TODO initial zero, trebuie sa incercam

        # Calculates the output of the decoder.
        output, attention_weights = self.decoder.forward(y, enc_output, dec_states, teacher_forcing_ratio)

        # Creates a BOS tensor that must be added to the beginning of the output. [batch_size, 1, dec_vocab_size]
        bos_tensor = torch.zeros(batch_size, 1, self.dec_vocab_size).to(self.device)
        # Marks the corresponding BOS position with a probability of 1.
        bos_tensor[:, :, 2] = 1

        # Concatenates the BOS tensor with the output. [batch_size, dec_seq_len-1, dec_vocab_size] -> [batch_size,
        # dec_seq_len, dec_vocab_size]
        output = torch.cat((bos_tensor, output), dim=1)

        return output, attention_weights

    def transfer_hidden_from_encoder_to_decoder(self, enc_states):
        batch_size = enc_states[0].shape[1]

        # Reshapes the shape of the hidden and cell state of the encoder LSTM layers. Permutes the batch_size to
        # the first dimension, and reshapes them to a 2-D tensor.
        # [enc_num_layers * 2, batch_size, enc_hidden_dim] -> [batch_size, enc_num_layers * enc_hidden_dim * 2].
        enc_states = (enc_states[0].permute(1, 0, 2).reshape(batch_size, -1),
                      enc_states[1].permute(1, 0, 2).reshape(batch_size, -1))

        # Transforms the hidden and the cell state of the encoder lstm layer to correspond to the decoder lstm states dimensions.
        # [batch_size, enc_num_layers * enc_hidden_dim * 2] -> [batch_size, dec_num_layers * dec_hidden_dim].
        dec_states = (self.h_state_linear(enc_states[0]), self.c_state_linear(enc_states[1]))

        # Reshapes the states to have the correct shape for the decoder lstm states dimension. Reshape the states from
        # 2-D to 3-D sequence. Permutes the batch_size to the second dimension.
        # [batch_size, dec_num_layers * dec_hidden_dim] -> [dec_num_layers, batch_size, dec_hidden_dim].
        dec_states = (dec_states[0].reshape(batch_size, self.dec_num_layers, self.dec_hidden_dim).permute(1, 0, 2),
                      dec_states[1].reshape(batch_size, self.dec_num_layers, self.dec_hidden_dim).permute(1, 0, 2))

        return dec_states

    def load_checkpoint(self, folder, extension):
        filename = os.path.join(folder, "checkpoint." + extension)
        print("Loading model {} ...".format(filename))
        if not os.path.exists(filename):
            print("\tModel file not found, not loading anything!")
            return {}

        checkpoint = torch.load(filename)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        return checkpoint["extra"]

    def save_checkpoint(self, folder, extension, extra={}):
        filename = os.path.join(folder, "checkpoint." + extension)
        checkpoint = {}
        checkpoint["encoder_state_dict"] = self.encoder.state_dict()
        checkpoint["decoder_state_dict"] = self.decoder.state_dict()
        checkpoint["extra"] = extra
        torch.save(checkpoint, filename)



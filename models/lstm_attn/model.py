import sys
sys.path.insert(0, '../..')

from models.components.decoders.LSTMDecoderWithAdditiveAttention import LSTMDecoderWithAdditiveAttention
from models.lstm.model import LSTMEncoderDecoder


class LSTMEncoderDecoderWithAdditiveAttention(LSTMEncoderDecoder):
    def __init__(self,
                 # encoder params
                 enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers, enc_dropout, enc_lstm_dropout,
                 # decoder params
                 dec_input_dim, dec_emb_dim, dec_hidden_dim, dec_num_layers, dec_dropout, dec_lstm_dropout, dec_vocab_size, dec_transfer_hidden=False):
        
        """
        Creates a simple Encoder-Decoder with additive attention.

        Args: #
            See LSTMEncoderDecoder for further args info.
        """

        super(LSTMEncoderDecoderWithAdditiveAttention, self).__init__(# encoder params
                                                                      enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers,
                                                                      enc_dropout, enc_lstm_dropout,
                                                                      # decoder params
                                                                      dec_input_dim, dec_emb_dim, dec_hidden_dim, dec_num_layers,
                                                                      dec_vocab_size, dec_transfer_hidden)

        self.decoder = LSTMDecoderWithAdditiveAttention(emb_dim=dec_emb_dim, input_size=enc_hidden_dim, hidden_dim=dec_hidden_dim, num_layers=dec_num_layers,
                                                        n_class=dec_vocab_size, lstm_dropout=dec_lstm_dropout, dropout=dec_dropout, device=self.device)

        self.to(self.device)

    
        
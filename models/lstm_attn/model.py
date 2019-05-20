import sys
sys.path.insert(0, '../..')

from models.components.RNNEncoderDecoder import RNNEncoderDecoder

from models.components.encoders.LSTMEncoder import LSTMEncoder
from models.components.decoders.LSTMDecoderWithAdditiveAttention import LSTMDecoderWithAdditiveAttention

class LSTMEncoderDecoderWithAdditiveAttention(RNNEncoderDecoder):
    def __init__(self,
                 # encoder params
                 enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers, enc_dropout, enc_lstm_dropout,
                 # decoder params
                 dec_input_dim, dec_emb_dim, dec_hidden_dim, dec_num_layers, dec_dropout, dec_lstm_dropout, dec_vocab_size, dec_transfer_hidden=False):
        
        """
        Creates a simple Encoder-Decoder with additive attention.

        Args: #
            See RNNEncoderDecoder for further args info.
        """
      
        super(LSTMEncoderDecoderWithAdditiveAttention, self).__init__(# encoder params
                                                                      LSTMEncoder, enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers,
                                                                      enc_dropout, enc_lstm_dropout,
                                                                      # decoder params
                                                                      LSTMDecoderWithAdditiveAttention, dec_input_dim, dec_emb_dim, dec_hidden_dim, dec_num_layers, dec_vocab_size, dec_lstm_dropout, dec_dropout, dec_transfer_hidden)

        
        self.to(self.device)

    
        
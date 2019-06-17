import sys
sys.path.insert(0, '../..')

from models.components.RNNEncoderDecoder import RNNEncoderDecoder

from models.components.encoders.LSTMSelfAttentionEncoder import LSTMSelfAttentionEncoder
from models.components.decoders.LSTMDecoderWithAttention import LSTMDecoderWithAttention

class LSTMEncoderDecoderWithAttentionAndSelfAttention(RNNEncoderDecoder):
    def __init__(self,
                 # encoder params
                 enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers, enc_lstm_dropout, enc_dropout,
                 # decoder params
                 dec_input_dim, dec_emb_dim, dec_hidden_dim, dec_num_layers, dec_lstm_dropout, dec_dropout, dec_vocab_size, dec_attention_type, dec_transfer_hidden=False):
        
        """
        Creates a Encoder-Decoder with additive attention and self-attention.

        Args: #
            See RNNEncoderDecoder for further args info.
        """
      
        super(LSTMEncoderDecoderWithAttentionAndSelfAttention, self).__init__(        
            # encoder params
            LSTMSelfAttentionEncoder, enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers, enc_lstm_dropout, enc_dropout,
            # decoder params
            LSTMDecoderWithAttention, dec_input_dim, dec_emb_dim, dec_hidden_dim, dec_num_layers, dec_vocab_size, dec_lstm_dropout, dec_dropout, dec_attention_type, dec_transfer_hidden)

        
        self.to(self.device)

    
        
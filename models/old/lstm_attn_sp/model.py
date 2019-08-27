import sys
sys.path.insert(0, '../..')

from models.components.RNNEncoderDecoder import RNNEncoderDecoder

from models.components.encoders.LSTMEncoder import LSTMEncoder
from models.components.decoders.LSTMDecoderWithAttentionAndScratchPad import LSTMDecoderWithAttentionAndScratchPad

class LSTMEncoderDecoderWithAttentionAndScratchPad(RNNEncoderDecoder):
    def __init__(self, src_lookup, tgt_lookup,
                 # encoder params
                 enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers, enc_lstm_dropout, enc_dropout,
                 # decoder params
                 dec_input_dim, dec_emb_dim, dec_hidden_dim, dec_num_layers, dec_lstm_dropout, dec_dropout, dec_vocab_size, dec_attention_type, dec_transfer_hidden=True):
       
        super(LSTMEncoderDecoderWithAttentionAndScratchPad, self).__init__( src_lookup, tgt_lookup,       
            # encoder params
            LSTMEncoder, enc_vocab_size, enc_emb_dim, enc_hidden_dim, enc_num_layers, enc_lstm_dropout, enc_dropout,
            # decoder params
            LSTMDecoderWithAttentionAndScratchPad, dec_input_dim, dec_emb_dim, dec_hidden_dim, dec_num_layers, dec_vocab_size, dec_lstm_dropout, dec_dropout, dec_attention_type, dec_transfer_hidden)
        
        self.to(self.device)

    
        
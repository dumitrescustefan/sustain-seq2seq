import sys, os
sys.path.insert(0, '../..')

from models.components.encoders.SimpleSlotEncoder import SimpleSlotEncoder
from models.components.decoders.LSTMDecoderWithAttentionAndSelfAttention import LSTMDecoderWithAttentionAndSelfAttention
import torch.nn as nn
import torch

class NLG_SimpleEncoder_LSTMDecoderWithAttentionAndSelfAttention(nn.Module):
    def __init__(self,
                 # encoder params
                 enc_emb_dim, slot_sizes, enc_dropout,
                 # decoder params
                 dec_input_dim, dec_emb_dim, dec_hidden_dim, dec_num_layers, dec_lstm_dropout, dec_dropout, dec_vocab_size, dec_attention_type):
      
        super(NLG_SimpleEncoder_LSTMDecoderWithAttentionAndSelfAttention, self).__init__()

        if torch.cuda.is_available():
            print('Running on GPU.')
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            print('Running on CPU.')
            self.cuda = False
            self.device = torch.device('cpu')

        self.enc_emb_dim = enc_emb_dim
        self.slot_sizes = slot_sizes               
        self.dec_input_dim = dec_input_dim
        self.dec_emb_dim = dec_emb_dim        
        self.dec_hidden_dim = dec_hidden_dim
        self.dec_num_layers = dec_num_layers        
        self.dec_vocab_size = dec_vocab_size
        self.dec_attention_type = dec_attention_type

        self.encoder = SimpleSlotEncoder(emb_dim = enc_emb_dim, slot_sizes = slot_sizes, dropout = enc_dropout, device=self.device)
        
        self.decoder = LSTMDecoderWithAttentionAndSelfAttention(emb_dim=dec_emb_dim, input_size=enc_emb_dim, hidden_dim=dec_hidden_dim, num_layers=dec_num_layers,
                            n_class=dec_vocab_size, lstm_dropout=dec_lstm_dropout, dropout=dec_dropout, attention_type=dec_attention_type, device=self.device)

        self.to(self.device)

    def forward(self, x, y, teacher_forcing_ratio=0.):
        """
        Args:
            x (tensor): The input of the decoder. Shape: [batch_size, seq_len_enc].
            y (tensor): The input of the decoder. Shape: [batch_size, seq_len_dec].

        Returns:
            Shape: [batch_size, seq_len_dec, n_class].
        """
        batch_size = x.shape[0]

        # Calculates the output of the encoder
        enc_output = self.encoder.forward(x)

        # (num_layers * num_directions, batch, hidden_size)
        h_0 = torch.zeros(self.dec_num_layers, batch_size, self.dec_hidden_dim, device = self.device)
        c_0 = torch.zeros(self.dec_num_layers, batch_size, self.dec_hidden_dim, device = self.device)
        dec_states = (h_0, c_0)
   
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
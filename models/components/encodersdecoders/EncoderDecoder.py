import os, sys

sys.path.insert(0, '../..')

import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, src_lookup, tgt_lookup, encoder, decoder, device):
        super().__init__()
        
        if torch.cuda.is_available():            
            self.cuda = True
            self.device = torch.device('cuda')
        else:            
            self.cuda = False
            self.device = torch.device('cpu')

        self.src_lookup = src_lookup
        self.tgt_lookup = tgt_lookup
        self.src_bos_token_id = src_lookup.convert_tokens_to_ids(src_lookup.bos_token)
        self.src_eos_token_id = src_lookup.convert_tokens_to_ids(src_lookup.eos_token)
        self.tgt_bos_token_id = src_lookup.convert_tokens_to_ids(tgt_lookup.bos_token)
        self.tgt_eos_token_id = src_lookup.convert_tokens_to_ids(tgt_lookup.eos_token)
    
        self.encoder = encoder       
        self.decoder = decoder
        
        self.device = device
        self.to(self.device)

    def forward(self, x_tuple, y_tuple, teacher_forcing_ratio=0.):
        raise Exception("forward() not implemented")
    
    def run_batch(self, X_tuple, y_tuple, criterion=None, tf_ratio=.0, aux_loss_weight = 0.5):
        raise Exception("run_batch() not implemented")

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



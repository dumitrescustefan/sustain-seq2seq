import os, sys
sys.path.insert(0, '../../..')

import torch
import torch.nn as nn

class ScratchPad(nn.Module):
    def __init__(self, enc_size, dec_size, device):
        """
        https://arxiv.org/pdf/1906.05275v2.pdf
        bupL https://arxiv.org/abs/1808.10792
        """
        super(ScratchPad, self).__init__()
        self.enc_size = enc_size
        self.dec_size = dec_size
        
        # to compute alpha we need the updated (i+1) state of the decoder, the current context (i) and encoder state (0:seq_len)
        self.mlp_alpha = nn.Linear(dec_size+enc_size+enc_size, 1)
        
        # to compute u we need need the updated (i+1) state of the decoder, the current context (i)
        self.mlp_u = nn.Linear(dec_size+enc_size, enc_size)
        
        from models.util.log import Log
        log_path = os.path.join("tensors")
        self.log_object = Log(log_path, clear=True)        
        self.plot_every = 1000
        self.forwards = 0
        
        self.device = device
        self.to(device)

    def _reshape_state_h(self, state_h):    
        """
        Reshapes the hidden state to desired shape
        Input: [num_layers * 1, batch_size, decoder_hidden_size]  
        Output: [batch_size, 1, decoder_hidden_state]

        Args:
            state_h (tensor): Hidden state of the decoder.        
                [num_layers * 1, batch_size, decoder_hidden_size]
        
        Returns:
            The reshaped hidden state.
                [batch_size, 1, decoder_hidden_state]
        """
        num_layers, batch_size, hidden_size = state_h.size()
        # in case the decoder has more than 1 layer, take only the last one -> [1, batch_size, decoder_hidden_size]
        if num_layers > 1:
            state_h = state_h[num_layers-1:num_layers,:,:]

        # [1, batch_size, decoder_hidden_size] -> [batch_size, 1, decoder_hidden_size]
        return state_h.permute(1, 0, 2)

    def forward(self, encoder_output, decoder_state, context, decoder_step = 0, plot = False):
        """
        Args:
            encoder_output: [batch_size, seq_len_enc, enc_size]
            decoder_state:  [num_layers, batch_size, dec_size] -> [batch_size, 1, dec_size] (we'll reshape and take top layer only)
            context:        [batch_size, enc_size]
        Returns:
            Rewritten encoder outputs:
                Shape:      [batch_size, seq_len_enc, enc_size]
        """
        batch_size = encoder_output.size()[0]
        seq_len = encoder_output.size()[1]
        
        # reshape state_h
        reshaped_decoder_state = self._reshape_state_h(decoder_state[0])
        
        # first calculate u as the tanh(mlp(state, context))
        input = torch.zeros(batch_size, self.dec_size+self.enc_size, device = self.device)
        input[:,:self.dec_size] = reshaped_decoder_state[:,0,:]
        input[:,self.dec_size:] = context[:,:]
        u = self.mlp_u(input).tanh()
        
        
        if self.forwards%self.plot_every == 0:
            plot = True
        if plot:
            self.log_object.plot_heatmaps(encoder_output, "encoder_output", epoch = self.forwards+decoder_step*2)
            
        # for each encoder output step i, compute alpha as sigmoid(mlp(state, context, encoder_output))
        input = torch.zeros(batch_size, self.dec_size+self.enc_size+self.enc_size, device = self.device)
        for i in range(seq_len):            
            input[:,:self.dec_size] = reshaped_decoder_state[:,0,:]
            input[:,self.dec_size:self.dec_size+self.enc_size] = context[:,:]
            input[:,self.dec_size+self.enc_size:] = context[:,:]
            alpha = self.mlp_alpha(input).sigmoid()
            
            # update encoder state i
            encoder_output[:,i:i+1,:] = alpha * encoder_output[:,i:i+1,:] + (1. - alpha) * u
        
        
        if plot:
            self.log_object.plot_heatmaps(encoder_output, "encoder_output", epoch = self.forwards+decoder_step*2+1)
        
        self.forwards+=1
        return encoder_output
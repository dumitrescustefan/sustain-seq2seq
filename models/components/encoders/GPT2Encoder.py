import os, sys
sys.path.insert(0, '../../..')

import torch
import torch.nn as nn
from pytorch_transformers import GPT2Model

class Encoder(nn.Module):
    def __init__(self, vocab_size, device):       
        super().__init__()
        
        self.gpt2model = GPT2Model.from_pretrained('gpt2')
        self.gpt2model.resize_token_embeddings(vocab_size)
        self.to(device)

    def forward(self, input_tuple):
        """
        Args:
            input_tuple (tenspr): The input of the encoder. On the first position it must be a 2-D tensor of integers, padded. The second is the lenghts of the first.
                Shape: ([batch_size, seq_len_enc], [batch_size], other)

        Returns:
            A tuple containing the output and the states of the last LSTM layer. The states of the LSTM layer is also a
            tuple that contains the hidden and the cell state, respectively . 
                Output shape: [batch_size, seq_len_enc, 768]
        
        """
        
        X, X_lengths = input_tuple[0], input_tuple[1]
        
        self.gpt2model.eval()        
        with torch.no_grad():
            #print() 
            #print(X.size())
            last_hidden_states = self.gpt2model(X)[0]            
            #print(last_hidden_states.size())
            
        
        return {'output':last_hidden_states}
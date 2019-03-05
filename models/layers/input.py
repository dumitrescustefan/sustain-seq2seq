import torch
import torch.nn as nn
import numpy as np

class InputLayerWithAbsolutePosition(nn.Module):
    """
       
    """
    def __init__(self, vocab_size, embedding_dim, max_seq_len=512): # transforms a batched padded input sequence in absolute positional embeddings
        super(InputLayerWithAbsolutePosition).__init__()
        
        self.embedding_dim = embedding_dim        
        self.max_seq_len = max_seq_len # this is only for positional embeddings to preinitialize only max_seq_len positions
        self.vocab_size = vocab_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                
        position_enc = np.array([ [pos / np.power(10000, 2 * (j // 2) / embedding_dim) for j in range(embedding_dim)] if pos != 0 else np.zeros(embedding_dim) for pos in range(max_seq_len+1)]) 
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) 
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) 
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.absolute_position_embedding = torch.nn.Embedding(max_seq_len+1, embedding_dim, padding_idx=0)
        self.absolute_position_embedding.weight.data = torch.FloatTensor(position_enc, device = self.device)        
        
    def forward(self, input_tensor, add_positional_encoding = True):
        """
        For encoder, process full sequence
            Input is (batch_size, max_seq_len) zero padded indexes
            Output is (batch_size, max_seq_len, embedding_dim)
        """        
        output_tensor = self.embedding(input_tensor)
        if not add_positional_encoding:
            return output_tensor
        
        # calculate lengths of input tensors
        lengths = [] 
        batch_size = input_tensor.size(0)
        for i in range(batch_size):
            li = np.arange(1, sequence_lenghts[i]+1, dtype=long)
            if sequence_lenghts[i]<self.max_seq_len: # pad with zeroes
                li += np.zeros(self.max_seq_len - sequence_lenghts[i], dtype=long)                
            lengths.append(li)
        lengths = torch.tensor(lengths, device = self.device)        
        # lengths is an array of (bs, seq_len) with each row as [1,2,3, k, 0,0..0] of len max_seq_len        
        return output_tensor + self.absolute_position_embedding(lengths) 
    
    def forward_step(self, input_tensor_element, add_positional_encoding = True): 
        """
        For decoder, process one step at a time
            Input is (batch_size, 1) containing index of word
            Output is (batch_size, 1, embedding_dim)
        """
        # todo in decode
        pass
 
import torch
import torch.nn as nn
import numpy as np

class InputLayerWithAbsolutePosition(nn.Module):
    """
       
    """
    def __init__(self, vocab_size, embedding_dim, max_seq_len=512): # transforms a batched padded input sequence in absolute positional embeddings
        super(InputLayerWithAbsolutePosition, self).__init__()
        
        self.embedding_dim = embedding_dim        
        self.max_seq_len = max_seq_len # this is only for positional embeddings to preinitialize only max_seq_len positions
        self.vocab_size = vocab_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                
        position_enc = np.array([ [pos / np.power(10000, 2 * (j // 2) / embedding_dim) for j in range(embedding_dim)] if pos != 0 else np.zeros(embedding_dim) for pos in range(max_seq_len+1)]) 
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) 
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) 
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.absolute_position_embedding = nn.Embedding(max_seq_len+1, embedding_dim, padding_idx=0)
        self.absolute_position_embedding.weight.data = torch.FloatTensor(position_enc, device = self.device)        
        self.absolute_position_embedding.weight.requires_grad = False
        # parameters = filter(lambda p: p.requires_grad, net.parameters())
        
    def forward(self, input_tensor, incremental_mask = None, add_positional_encoding = True):
        """
        For encoder, process full sequence
            Input is (batch_size, max_seq_len) zero padded indexes
            Output is (batch_size, max_seq_len, embedding_dim)
        """        
        output_tensor = self.embedding(input_tensor)
        if not add_positional_encoding:
            return output_tensor
           
        if incremental_mask is None:            
            _, incremental_mask = self.get_mask(input_tensor)
        
        # lengths is an array of (bs, seq_len) with each row as [1,2,3, k, 0,0..0] of len max_seq_len        
        return output_tensor + self.absolute_position_embedding(incremental_mask) 
    
    def forward_step(self, input_tensor_element, step_index, add_positional_encoding = True): 
        """
        For decoder, process one step at a time
            Input is (batch_size, 1) containing index of word
            Output is (batch_size, 1, embedding_dim)
        """
        # todo in decode
        pass
    
    
    def get_mask (self, input_tensor):
        """
            input_tensor is a (batch_size, seq_len) 0-padded tensor
            output a (batch_size, seq_len) with 1,2,3,k,0,0,0   of seq_len and 1,1,1,1..,0,0,0 
        """
        incremental_mask = []
        attention_mask = []
        batch_size = input_tensor.size(0)
        seq_len = input_tensor.size(1)        
        for i in range(batch_size):
            # search backwards for non-zero element
            j = seq_len-1            
            #print(input_tensor[i][j])
            while input_tensor[i][j] == 0. and j>=0:
                j-=1
            li = np.arange(1, j+2, dtype=np.long)
            la = np.ones(j+1, dtype=np.long)
            if j<self.max_seq_len-1: # pad with zeroes
                li = np.concatenate((li, np.zeros(seq_len - j-1, dtype=np.long)), axis=0)
                la = np.concatenate((la, np.zeros(seq_len - j-1, dtype=np.long)), axis=0)
            incremental_mask.append(li)
            attention_mask.append(la)
        return torch.tensor(attention_mask, dtype=torch.long, device = self.device), torch.tensor(incremental_mask, dtype=torch.long, device = self.device)
        
    
if __name__ == '__main__':  
    inp = InputLayerWithAbsolutePosition(vocab_size=8096, embedding_dim=2, max_seq_len=512)
    input_tensor = torch.tensor([[1,23,43,0,0,0],[0,0,0,0,0,0],[4,4,4,4,4,4]],dtype=torch.long) #ones(4, 10, dtype=torch.long)
    attention_mask, incremental_mask = inp.get_mask(input_tensor)
    print(attention_mask)
    print(incremental_mask)
    output_tensor = inp(input_tensor, add_positional_encoding = True)
    print(output_tensor)
    
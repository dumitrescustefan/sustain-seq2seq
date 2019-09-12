import sys, os
sys.path.insert(0, '../..')

from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.stats
from models.components.encodersdecoders.EncoderDecoder import EncoderDecoder
from pytorch_transformers import GPT2Model

class MiniClassifier(nn.Module):
    def __init__(self, vocab_size, dec_hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size                
        self.slot_linear = nn.Linear(dec_hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim = 1)
        self.criterion = nn.CrossEntropyLoss()#nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, x, slots, my_index): 
        target = slots[:,my_index]
        #print("target is : ")
        #print(target)
       
        logits = self.slot_linear(x) # [batch_size, dec_vocab_size]
        #x = self.softmax(x) # [batch_size, dec_vocab_size]
        #print(x.size())
        #print("loss: ")
        #print(slots[:,my_index:my_index+1].size()) 
        
        
        #print(logits.size())
        #print(target.size())
        #print(logits)
        #print(target)
        #input("asd")
        loss = self.criterion(logits, target)
        #print(loss.item()) 
        return logits, loss



class MyEncoderDecoder(EncoderDecoder):
    def __init__(self, src_lookup, tgt_lookup, encoder, decoder, slot_sizes, dec_transfer_hidden, coverage_loss_weight, attention_loss_weight, device):
        super().__init__(src_lookup, tgt_lookup, encoder, decoder, device)

        self.coverage_loss_weight = coverage_loss_weight
        self.attention_loss_weight = attention_loss_weight
        self.dec_transfer_hidden = dec_transfer_hidden
        self.current_epoch = 0 
        self.start_slot_loss_epoch = 2 # start calculating loss after this epoch 
        
        if dec_transfer_hidden == True:
            assert encoder.num_layers == decoder.num_layers, "For transferring the last hidden state from encoder to decoder, both must have the same number of layers."

        # Transform h from encoder's [num_layers * 2, batch_size, enc_hidden_dim/2] to decoder's [num_layers * 1, batch_size, dec_hidden_dim], same for c; batch_size = 1 (last timestep only)
        self.h_state_linear = nn.Linear(int(encoder.hidden_dim * encoder.num_layers/1), decoder.hidden_dim * decoder.num_layers * 1)
        self.c_state_linear = nn.Linear(int(encoder.hidden_dim * encoder.num_layers/1), decoder.hidden_dim * decoder.num_layers * 1)

        self.attention_criterion = nn.KLDivLoss(reduction='batchmean')
        
        # slot values prediction
        self.slot_sizes = slot_sizes        
                
        self.gpt2model = GPT2Model.from_pretrained('gpt2')
        self.gpt2model.resize_token_embeddings(len(tgt_lookup))
            
        self.slot_pre_dropout = nn.Dropout(decoder.linear_dropout)
        self.slot_rnn = nn.LSTM(768, int(decoder.hidden_dim/2), num_layers=decoder.num_layers, dropout=decoder.lstm_dropout, bidirectional=True, batch_first=True)
        self.slot_post_dropout = nn.Dropout(decoder.linear_dropout)
        
        slot_networks = []                                
        for i in range(len(self.slot_sizes)): # how many slots there are            
            mc = MiniClassifier(vocab_size=self.slot_sizes[i], dec_hidden_dim=decoder.hidden_dim)
            slot_networks.append(mc)            
        self.slot_networks = nn.ModuleList(slot_networks)
        
        self.to(self.device)

    def start_train_epoch (self, current_epoch):
        self.current_epoch = current_epoch
        self.slot_cfmatrix = []
        for i in range(len(self.slot_sizes)):
            self.slot_cfmatrix.append(torch.zeros((self.slot_sizes[i], self.slot_sizes[i]), device = self.device))
     
    def end_train_epoch (self):   
        print()
        overall_acc = 0
        for i in range(len(self.slot_sizes)):
            cf = self.slot_cfmatrix[i]
            vocab_size = self.slot_sizes[i]
            acc = 0
            diag_sum = torch.sum(cf.diag())
            total_sum = torch.sum(cf)
            acc = (diag_sum/total_sum).item()
            overall_acc+=acc
            print("Slot index {} has accuracy {:.2f}".format(i, acc))
            """
            for j in range(vocab_size):
                for q in range(vocab_size):
                    val = int(cf[j,q].item())
                    if val == 0 :
                        val = "•" #◦
                    else:
                        val = str(val)
                    print(val.rjust(4, ' '), end='')
                print()
            """
        print("Slot accuracy: {:.3f}".format(overall_acc/len(self.slot_sizes)))      

    def forward(self, x_tuple, y_tuple, teacher_forcing_ratio=0.):
        """
        Args:
            x (tensor): The input of the decoder. Shape: [batch_size, seq_len_enc].
            y (tensor): The input of the decoder. Shape: [batch_size, seq_len_dec].

        Returns:
            The output of the Encoder-Decoder with attention. Shape: [batch_size, seq_len_dec, n_class].
        """
        x, x_lenghts, x_mask = x_tuple[0], x_tuple[1], x_tuple[2]
        batch_size = x.shape[0]
        
        # Calculates the output of the encoder
        encoder_dict = self.encoder.forward(x_tuple)
        enc_output = encoder_dict["output"]
        enc_states = encoder_dict["states"]
        # enc_states is a tuple of size ( h=[enc_num_layers*2, batch_size, enc_hidden_dim/2], c=[same-as-h] )

        if self.dec_transfer_hidden == True:
            dec_states = self.transfer_hidden_from_encoder_to_decoder(enc_states)
        else:
            hidden = Variable(next(self.parameters()).data.new(batch_size, self.decoder.num_layers, self.decoder.hidden_dim), requires_grad=False)
            cell = Variable(next(self.parameters()).data.new(batch_size, self.decoder.num_layers, self.decoder.hidden_dim), requires_grad=False)
            dec_states = ( hidden.zero_().permute(1, 0, 2), cell.zero_().permute(1, 0, 2) )

        # Calculates the output of the decoder.
        decoder_dict = self.decoder.forward(x_tuple, y_tuple, enc_output, dec_states, teacher_forcing_ratio)
        output = decoder_dict["output"]
        attention_weights = decoder_dict["attention_weights"]
        coverage_loss = decoder_dict["coverage_loss"]
        hidden_states = decoder_dict["hidden_states"] # list of tuples of ([num_layers * num_directions, batch, hidden_size], same)
        # Creates a BOS tensor that must be added to the beginning of the output. [batch_size, 1, dec_vocab_size]
        bos_tensor = torch.zeros(batch_size, 1, self.decoder.vocab_size).to(self.device)
        # Marks the corresponding BOS position with a probability of 1.
        bos_tensor[:, :, self.tgt_bos_token_id] = 1
        # Concatenates the BOS tensor with the output. [batch_size, dec_seq_len-1, dec_vocab_size] -> [batch_size, dec_seq_len, dec_vocab_size]
        
        output = torch.cat((bos_tensor, output), dim=1)

        # now run slots based on output
        slot_loss = 0        
        if y_tuple is not None and self.current_epoch > self.start_slot_loss_epoch:
            y, y_lenghts, y_mask, slots = y_tuple[0], y_tuple[1], y_tuple[2], y_tuple[3]
            
            slot_inputs = torch.squeeze(torch.argmax(output, dim=2), dim=1) # [batch_size, dec_seq_len]            
            self.gpt2model.eval()            
            slot_encodings = torch.zeros(batch_size, seq_len, self.hidden_size).to(self.device)
            slot_encodings.requires_grad = False
        
            hidden_states, past = self.gpt2model(slot_inputs)  
            for i in range(batch_size):
                slot_encodings[i:i+1, 0:x_lengths[i], :] = hidden_states[i:i+1, 0:x_lengths[i], :]
            x = self.slot_pre_dropout(slot_encodings) #[batch_size, dec_seq_len, dec_vocab_size]
            x, _ = self.slot_rnn(x) #[batch_size, dec_seq_len, dec_hidden_dim]
            #print("after rnn")
            #print(x.size())
            x = x[:, -1, :] # last state is [batch_size, dec_hidden_dim]        
            #print("after last state")
            #print(x.size())
            x = self.slot_post_dropout(x) # [batch_size, dec_hidden_dim]
            #print("after dropout")
            #print(x.size())
                        
            for my_index, slot_network in enumerate(self.slot_networks):
                #print(slot_network)
                slot_logits, s_loss = slot_network(x, slots, my_index) # [batch_size, len(self.slot_sizes[my_index])]
                slot_loss += s_loss
                
                # update cf matrix
                my_prediction = torch.argmax(slot_logits, dim = 1) # [batch_size]
                target = slots[:,my_index] # [batch_size]
                for b in range(batch_size):                    
                    self.slot_cfmatrix[my_index][target[b], my_prediction[b]]+=1
                
            slot_loss /= len(self.slot_networks)
            
        return output, attention_weights, coverage_loss, slot_loss
     
    def run_batch(self, X_tuple, y_tuple=None, criterion=None, tf_ratio=.0):  
        # train mode: X, Y, criterion and tf_ratio are supplied
        # eval mode: X, Y, criterion are supplied, tf_ratio is default 0
        # run mode: X is supplied, Y and tf_ratio are None, tf_ratio is default 0 
        (x_batch, x_batch_lenghts, x_batch_mask) = X_tuple
        
        if y_tuple is not None:
            (y_batch, y_batch_lenghts, y_batch_mask, slots) = y_tuple
        
        if hasattr(self.decoder.attention, 'init_batch'):
            self.decoder.attention.init_batch(x_batch.size()[0], x_batch.size()[1])
        
        output, attention_weights, coverage_loss, slot_loss = self.forward(X_tuple, y_tuple, tf_ratio)
        
        display_variables = OrderedDict()
        
        disp_total_loss = 0
        disp_gen_loss = 0
        disp_cov_loss = 0
        disp_att_loss = 0 
        disp_slt_loss = 0
        total_loss = 0
        
        if criterion is not None:            
            gen_loss = criterion(output.view(-1, self.decoder.vocab_size), y_batch.contiguous().flatten())        
            disp_gen_loss = gen_loss.item()            
            total_loss = gen_loss + self.coverage_loss_weight*coverage_loss        
            disp_cov_loss = self.coverage_loss_weight*coverage_loss.item()
            
            #print("\nloss {:.3f}, aux {:.3f}*{}={:.3f}, total {}\n".format( loss, coverage_loss, coverage_loss_weight, coverage_loss_weight*coverage_loss, total_loss))
            if self.current_epoch > self.start_slot_loss_epoch:
                total_loss += slot_loss
                disp_slt_loss = slot_loss.item()
                    
            if tf_ratio>.0: # additional loss for attention distribution , attention_weights is [batch_size, seq_len] and is a list              
                batch_size = attention_weights.size(0)
                dec_seq_len = attention_weights.size(1)
                enc_seq_len = attention_weights.size(2)
                
                x = np.linspace(0, enc_seq_len, enc_seq_len)
                
                # create target distribution
                target_attention_distribution = attention_weights.new_full((batch_size, dec_seq_len, enc_seq_len), 1e-31)
                for decoder_index in range(0, dec_seq_len):
                    y = scipy.stats.norm.pdf(x, decoder_index, 2) # loc (mean) is decoder_step, scale (std dev) = 1.        
                    y = y / np.sum(y) # rescale to make it a PDF
                    gaussian_dist = torch.tensor(y, dtype = attention_weights.dtype, device = self.device) # make it a tensor, it's [seq_len]
                    target_attention_distribution[:,decoder_index, :] = gaussian_dist.repeat(batch_size, 1) # same for all examples in batch, now it's [batch_size, seq_len]
                
                target_attention_distribution[target_attention_distribution<1e-31] = 1e-31
                attention_weights[attention_weights<1e-31] = 1e-31
                
                #print(target_attention_distribution[0,0,:])                                
                #print(attention_weights_tensor[0,0,:])                
                
                attention_loss = tf_ratio * self.attention_criterion(target_attention_distribution.log().permute(0,2,1), attention_weights.permute(0,2,1)) * self.attention_loss_weight
                disp_att_loss = attention_loss.item()
                total_loss += attention_loss     
        
            display_variables["gen_loss"] = disp_gen_loss
            display_variables["cov_loss"] = disp_cov_loss
            display_variables["att_loss"] = disp_att_loss
            display_variables["slt_loss"] = disp_slt_loss
            
            
        return output, total_loss, attention_weights, display_variables
        
    def transfer_hidden_from_encoder_to_decoder(self, enc_states):
        batch_size = enc_states[0].shape[1]

        # Reshapes the shape of the hidden and cell state of the encoder LSTM layers. Permutes the batch_size to
        # the first dimension, and reshapes them to a 2-D tensor.
        # [enc_num_layers * 2, batch_size, enc_hidden_dim] -> [batch_size, enc_num_layers * enc_hidden_dim * 2].
        enc_states = (enc_states[0].permute(1, 0, 2).reshape(batch_size, -1),
                      enc_states[1].permute(1, 0, 2).reshape(batch_size, -1))

        # Transforms the hidden and the cell state of the encoder lstm layer to correspond to the decoder lstm states dimensions.
        # [batch_size, enc_num_layers * enc_hidden_dim * 2] -> [batch_size, dec_num_layers * dec_hidden_dim].
        dec_states = (torch.tanh(self.h_state_linear(enc_states[0])), torch.tanh(self.c_state_linear(enc_states[1])))

        # Reshapes the states to have the correct shape for the decoder lstm states dimension. Reshape the states from
        # 2-D to 3-D sequence. Permutes the batch_size to the second dimension.
        # [batch_size, dec_num_layers * dec_hidden_dim] -> [dec_num_layers, batch_size, dec_hidden_dim].
        dec_states = (dec_states[0].reshape(batch_size, self.decoder.num_layers, self.decoder.hidden_dim).permute(1, 0, 2),
                      dec_states[1].reshape(batch_size, self.decoder.num_layers, self.decoder.hidden_dim).permute(1, 0, 2))

        return dec_states


    
        
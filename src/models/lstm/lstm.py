from pprint import pprint

import torch
import torch.nn as nn

from layers import SimpleLSTMEncoderLayer, SimpleLSTMDecoderLayer

    
class LSTMEncoderDecoder(nn.Module):
    def __init__(self, w2i, i2w, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, encoder_n_layers, decoder_n_layers, encoder_drop_prob=0.5, decoder_drop_prob=0.5, lr = 0.01):
        super(LSTMEncoderDecoder, self).__init__()
        
        self.encoder_hidden_dim = encoder_hidden_dim
        
        self.encoder = SimpleLSTMEncoderLayer(len(w2i), embedding_dim, encoder_hidden_dim, encoder_n_layers, encoder_drop_prob)
        self.decoder = SimpleLSTMDecoderLayer(len(w2i), embedding_dim, encoder_hidden_dim, decoder_hidden_dim, decoder_n_layers, decoder_drop_prob)
        
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        
        self.w2i = w2i
        self.i2w = i2w
        
        self.train_on_gpu=torch.cuda.is_available()        
        if(self.train_on_gpu):
            print('Training on GPU.')
        else:
            print('No GPU available, training on CPU.')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, train_loader, valid_loader, test_loader, batch_size):            
        
        input = torch.randn(3, 2, 2, requires_grad=True)                
        print(input)
        print(input.size())
        output = input.squeeze(1)
        print(output)
        print(output.size())
        
        import numpy as np
        input = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        print(input)
        print(input.size())
        output = input[:,1]
        print(output)
        print(output.size())
        """ :,n selects the n dimension
        tensor([[1, 2, 3],
                [4, 5, 6]])
        torch.Size([2, 3])
        tensor([2, 5])
        torch.Size([2])
        """
        
        #return
        print("\nStart training ...")                
       
        criterion = nn.CrossEntropyLoss()
        
        # training params
        epochs = 40 

        counter = 0
        print_every = 100
        clip=5 # gradient clipping
                
        # move model to GPU, if available
        if(self.train_on_gpu):
            self.encoder.cuda()
            self.decoder.cuda()

        # train for some number of epochs
        for e in range(epochs):   
            print("Epoch "+str(e))
            encoder_hidden = self.encoder.init_hidden(batch_size)
            decoder_hidden = self.decoder.init_hidden(batch_size)

            # batch loop            
            counter = 0 
            self.encoder.train()
            self.decoder.train()
            for x, y in train_loader: 
                counter += 1                
                max_seq_len_x = x.size(1)
                max_seq_len_y = y.size(1)
                loss = 0
                print("  Epoch {}, batch: {}, max_seq_len_x: {}, max_seq_len_y: {}".format(e, counter, max_seq_len_x, max_seq_len_y))
                if x.size(0) != batch_size:
                    print("\t Incomplete batch, skipping.")
                    continue
                # print(x.size()) # x is a 64 * 399 tensor (batch*max_seq_len_x)               

                if(self.train_on_gpu):
                    x, y = inputs.x(), y.cuda()
                
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                encoder_hidden = tuple([each.data for each in encoder_hidden])
                decoder_hidden = tuple([each.data for each in decoder_hidden])

                # zero grads in optimizer
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                
                # encoder
                # x is batch_size * max_seq_len_x
                encoder_output, encoder_hidden = self.encoder(x, encoder_hidden) 
                
                # encoder_output is batch_size * max_seq_len_x * encoder_hidden
                #print(encoder_output.size())
                
                # take last state of encoder as context # not necessary when using attention                
                context = torch.zeros(batch_size, 1, self.encoder_hidden_dim)
                for j in range(batch_size):
                    for k in range(max_seq_len_x-1,0,-1):                        
                        if x[j][k]!=0:
                            #print("Found at "+str(k))
                            context[j][0] = encoder_output[j][k]
                            break
                
                # context is last state of the encoder batch_size * encoder_hidden_dim
                
                loss = 0                 
                for i in range(max_seq_len_y): # why decoder_hidden is initialized in epoch and not in batch??
                    print("\t Decoder step {}/{}".format(i, max_seq_len_y))    
                    
                    # force correct input in next step
                    decoder_input = torch.zeros(batch_size, 1, dtype = torch.long) # 1 in middle is because lstm expects (batch, seq_len, input_size): 
                    for j in range(batch_size):
                        decoder_input[j]=y[j][i]                        
                    #print(decoder_input.size())
                    _, decoder_hidden, word_softmax_projection = self.decoder.forward_step(decoder_input, decoder_hidden, context)
                    # first, reduce word_softmax_projection which is torch.Size([64, 1, 50004]) to 64 * 50004
                    word_softmax_projection = word_softmax_projection.squeeze(1) # eliminate dim 1
                    
                    #word_softmax_projection_array.append(word_softmax_projection)
                    
                    # now, select target y
                    # y looks like batch_size * max_seq_len_y : tensor([[    2, 10890, 48108,  ...,     0,     0,     0], ... ... ..
                    target_y = y[:,i] # select from y the ith column and shape as an array 
                    # target_y now looks like [ 10, 2323, 5739, 24, 9785 ... ] of size 64 (batch_size)
                    #print(word_softmax_projection.size())
                    #print(target_y.size())
                    loss += criterion(word_softmax_projection, target_y)   # ignore index ??  
                 
                """ input, target
                tensor([[ 0.2375, -0.1487, -0.0832, -0.7523, -0.5504],
                        [ 0.6835, -0.4430,  0.3776, -0.4433, -0.2840],
                        [-0.3039,  2.1489, -1.7675, -0.9361, -0.3278]], requires_grad=True)
                tensor([4, 0, 3])
                criterion = nn.CrossEntropyLoss()
                criterion(input, target)
                """
                #print(y)
               
                # detach??
                
                # calculate the loss and perform backprop
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
                nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                print("\t\t\tLoss: {}".format(loss.data.item()))
                
            # EVALUATION ####################################################################
            self.encoder.eval()
            self.decoder.eval()
            encoder_hidden = self.encoder.init_hidden(batch_size)
            decoder_hidden = self.decoder.init_hidden(batch_size)

            counter = 0 
            for x, y in valid_loader:
                counter += 1                
                max_seq_len_x = x.size(1)
                max_seq_len_y = y.size(1)
                loss = 0
                print("  Epoch {}, eval batch: {}, max_seq_len_x: {}, max_seq_len_y: {}".format(e, counter, max_seq_len_x, max_seq_len_y))
                if x.size(0) != batch_size:
                    print("\t Incomplete batch, skipping.")
                    continue
                # print(x.size()) # x is a 64 * 399 tensor (batch*max_seq_len_x)               

                if(self.train_on_gpu):
                    x, y = inputs.x(), y.cuda()
                
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                encoder_hidden = tuple([each.data for each in encoder_hidden])
                decoder_hidden = tuple([each.data for each in decoder_hidden])

                encoder_output, encoder_hidden = self.encoder(x, encoder_hidden) 
                
                # encoder_output is batch_size * max_seq_len_x * encoder_hidden
                #print(encoder_output.size())
                
                # take last state of encoder as context # not necessary when using attention                
                context = torch.zeros(batch_size, 1, self.encoder_hidden_dim)
                for j in range(batch_size):
                    for k in range(max_seq_len_x-1,0,-1):                        
                        if x[j][k]!=0:
                            #print("Found at "+str(k))
                            context[j][0] = encoder_output[j][k]
                            break
                
                # context is last state of the encoder batch_size * encoder_hidden_dim
                
                loss = 0                 
                for i in range(max_seq_len_y): # why decoder_hidden is initialized in epoch and not in batch??
                    print("\t Decoder step {}/{}".format(i, max_seq_len_y))    
                    
                    # force correct input in next step
                    decoder_input = torch.zeros(batch_size, 1, dtype = torch.long) # 1 in middle is because lstm expects (batch, seq_len, input_size): 
                    for j in range(batch_size):
                        decoder_input[j]=y[j][i]                        
                    #print(decoder_input.size())
                    _, decoder_hidden, word_softmax_projection = self.decoder.forward_step(decoder_input, decoder_hidden, context)
                    # first, reduce word_softmax_projection which is torch.Size([64, 1, 50004]) to 64 * 50004
                    word_softmax_projection = word_softmax_projection.squeeze(1) # eliminate dim 1
                    
                    #word_softmax_projection_array.append(word_softmax_projection)
                    
                    # now, select target y
                    # y looks like batch_size * max_seq_len_y : tensor([[    2, 10890, 48108,  ...,     0,     0,     0], ... ... ..
                    target_y = y[:,i] # select from y the ith column and shape as an array 
                    # target_y now looks like [ 10, 2323, 5739, 24, 9785 ... ] of size 64 (batch_size)
                    #print(word_softmax_projection.size())
                    #print(target_y.size())
                    loss += criterion(word_softmax_projection, target_y)   # ignore index ??  
             
                
    def load_checkpoint(filename, enc = None, dec = None):
        print("loading model...")
        checkpoint = torch.load(filename)
        if enc:
            enc.load_state_dict(checkpoint["encoder_state_dict"])
        if dec:
            dec.load_state_dict(checkpoint["decoder_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
        return epoch

    def save_checkpoint(filename, enc, dec, epoch, loss, time):
        print("epoch = %d, loss = %f, time = %f" % (epoch, loss, time))
        if filename and enc and dec:
            print("saving model...")
            checkpoint = {}
            checkpoint["encoder_state_dict"] = enc.state_dict()
            checkpoint["decoder_state_dict"] = dec.state_dict()
            checkpoint["epoch"] = epoch
            checkpoint["loss"] = loss
            torch.save(checkpoint, filename + ".epoch%d" % epoch)
            print("saved model at epoch %d" % epoch)                      
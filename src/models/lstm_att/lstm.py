from pprint import pprint
import time, io, os, random

import torch
import torch.nn as nn

from layers import SimpleLSTMEncoderLayer, SimpleLSTMDecoderLayer, AttentionLayer

from tensorboardX import SummaryWriter

class LSTMEncoderDecoderAtt(nn.Module):
    def __init__(self, w2i, i2w, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, encoder_n_layers, decoder_n_layers, encoder_drop_prob=0.5, decoder_drop_prob=0.5, lr = 0.01, teacher_forcing_ratio=0.5, gradient_clip = 5):
        super(LSTMEncoderDecoderAtt, self).__init__()
        
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_n_layers = decoder_n_layers
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.gradient_clip = gradient_clip
        
        self.encoder = SimpleLSTMEncoderLayer(len(w2i), embedding_dim, encoder_hidden_dim, encoder_n_layers, encoder_drop_prob)
        self.decoder = SimpleLSTMDecoderLayer(len(w2i), embedding_dim, encoder_hidden_dim*2, decoder_hidden_dim, decoder_n_layers, decoder_drop_prob)
        self.attention = AttentionLayer(encoder_hidden_dim*2, decoder_hidden_dim) # *2 because encoder is bidirectional an thus hidden is double 
        
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters())+list(self.attention.parameters()), lr=lr)        
        self.criterion = nn.CrossEntropyLoss()
        
        self.w2i = w2i
        self.i2w = i2w
        self.epoch = 0
        
        self.train_on_gpu=torch.cuda.is_available()        
        if(self.train_on_gpu):
            print('Training on GPU.')
        else:
            print('No GPU available, training on CPU.')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.writer = SummaryWriter('/work/tmp')
        
    def show_tensor(x, prediction=None, source=None):
        fig = plt.figure(figsize=(12, 6))
        sns.heatmap(x,cmap="rainbow")
        plt.tight_layout()        
        return fig
            
            
    def train(self, train_loader, valid_loader, test_loader, batch_size):                   
        counter = 0        
        
        unique_counter = 0        
        # move model to GPU, if available
        if(self.train_on_gpu):
            self.encoder.cuda()
            self.decoder.cuda()
            self.attention.cuda()

        while True:                  
            loss = self._train_epoch(train_loader, batch_size)            
            self.save_checkpoint(".", self.epoch)
            
            loss = self._eval(valid_loader, batch_size)
            
            #self.load_checkpoint(".", latest = True)    
            
    def _train_epoch(self, train_loader, batch_size):                       
        self.epoch += 1
        self.encoder.train()
        self.decoder.train()
        self.attention.train()        
        
        encoder_hidden = self.encoder.init_hidden(batch_size)
        decoder_hidden = self.decoder.init_hidden(batch_size)

        for counter, (x, y) in enumerate(train_loader):             
            #if counter > 1:
            #    break                
            max_seq_len_x = x.size(1) # x este 64 x 399 (variabil)
            max_seq_len_y = y.size(1) # y este 64 x variabil
            loss = 0
            print("  Epoch {}, batch: {}/{}, max_seq_len_x: {}, max_seq_len_y: {}".format(self.epoch, counter, len(train_loader), max_seq_len_x, max_seq_len_y))
            if x.size(0) != batch_size:
                print("\t Incomplete batch, skipping.")
                continue
            # print(x.size()) # x is a 64 * 399 tensor (batch*max_seq_len_x)               

            if(self.train_on_gpu):
                x, y = x.cuda(), y.cuda()
            
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            encoder_hidden = tuple([each.data for each in encoder_hidden])
            decoder_hidden = tuple([each.data for each in decoder_hidden])
            #print(decoder_hidden[0].size())
            
            # zero grads in optimizer
            self.optimizer.zero_grad()                
            
            # encoder
            # x is batch_size x max_seq_len_x
            encoder_output, encoder_hidden = self.encoder(x, encoder_hidden)             
            # encoder_output is batch_size x max_seq_len_x x encoder_hidden
            #print(encoder_output.size())
            
            # create first decoder output for initial attention call, extract from decoder_hidden
            decoder_output = decoder_hidden[0].view(self.decoder_n_layers, 1, batch_size, self.decoder_hidden_dim) #torch.Size([2, 1, 64, 512])
            # it should look like batch_size x 1 x decoder_hidden_size, so tranform it
            decoder_output = decoder_output[-1].permute(1,0,2) 
            #print(decoder_output.size())
                
            loss = 0                 
            for i in range(max_seq_len_y): # why decoder_hidden is initialized in epoch and not in batch??
                #print("\t Decoder step {}/{}".format(i, max_seq_len_y))    
                
                # teacher forcing (or it is first word which always is start-of-sentence)
                if random.random()<=self.teacher_forcing_ratio or i==0:
                    decoder_input = torch.zeros(batch_size, 1, dtype = torch.long) # 1 in middle is because lstm expects (batch, seq_len, input_size): 
                    for j in range(batch_size):
                        decoder_input[j]=y[j][i]                
                    #print(decoder_input.size()) # batch_size x 1                            
                else: # feed own previous prediction extracted from word_softmax_projection
                    _, decoder_input = word_softmax_projection.max(1) # no need for values, just indexes 
                    decoder_input = decoder_input.unsqueeze(1) # from batch_size to batch_size x 1                    
                    #print(decoder_input.size()) # batch_size x 1                            
                    
                context = self.attention(encoder_output, decoder_output)
                # context is batch_size * encoder_hidden_dim
            
                decoder_output, decoder_hidden, word_softmax_projection = self.decoder.forward_step(decoder_input, decoder_hidden, context)
                # first, reduce word_softmax_projection which is torch.Size([64, 1, 50004]) to 64 * 50004
                word_softmax_projection = word_softmax_projection.squeeze(1) # eliminate dim 1
                
                #word_softmax_projection_array.append(word_softmax_projection)
                
                # now, select target y
                # y looks like batch_size * max_seq_len_y : tensor([[    2, 10890, 48108,  ...,     0,     0,     0], ... ... ..
                target_y = y[:,i] # select from y the ith column and shape as an array 
                # target_y now looks like [ 10, 2323, 5739, 24, 9785 ... ] of size 64 (batch_size)
                #print(word_softmax_projection.size())
                #print(target_y.size())
                loss += self.criterion(word_softmax_projection, target_y)   # ignore index not set as we want 0 to count to error too
            
            loss.backward() # calculate the loss and perform backprop
            
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.gradient_clip)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), self.gradient_clip)
            nn.utils.clip_grad_norm_(self.attention.parameters(), self.gradient_clip)
            self.optimizer.step()
            
            print("\tLoss: {}".format(loss.data.item()))                
            self.writer.add_scalar('Train/Loss', loss.data.item())            
            break

    def _eval(self, valid_loader, batch_size):                
        start_time = time.time()
        self.encoder.eval()
        self.decoder.eval()
        self.attention.eval()            
        encoder_hidden = self.encoder.init_hidden(batch_size)
        decoder_hidden = self.decoder.init_hidden(batch_size)

        counter = 0 
        with torch.no_grad():
            for counter, (x, y) in enumerate(valid_loader):
                counter += 1  
                if counter > 5:
                    break
                max_seq_len_x = x.size(1)
                max_seq_len_y = y.size(1)
                loss = 0
                print("  Epoch {}, batch: {}/{}, max_seq_len_x: {}, max_seq_len_y: {}".format(self.epoch, counter, len(valid_loader), max_seq_len_x, max_seq_len_y))
                if x.size(0) != batch_size:
                    print("\t Incomplete batch, skipping.")
                    continue
                
                if(self.train_on_gpu):
                    x, y = x.cuda(), y.cuda()
                
                encoder_hidden = tuple([each.data for each in encoder_hidden])
                decoder_hidden = tuple([each.data for each in decoder_hidden])

                encoder_output, encoder_hidden = self.encoder(x, encoder_hidden) 
                word_softmax_projection = torch.zeros(batch_size, 3, dtype = torch.float)
                word_softmax_projection[:,2] = 1. # beginning of sentence value is 2, set it 
                decoder_output = decoder_hidden[0].view(self.decoder_n_layers, 1, batch_size, self.decoder_hidden_dim) #torch.Size([2, 1, 64, 512])
                decoder_output = decoder_output[-1].permute(1,0,2) 
                                
                loss = 0             
                print_example = True
                example_array = []
                
                for i in range(max_seq_len_y): # why decoder_hidden is initialized in epoch and not in batch??
                    #print("\t Decoder step {}/{}".format(i, max_seq_len_y))                        
                    _, decoder_input = word_softmax_projection.max(1) # no need for values, just indexes 
                    decoder_input = decoder_input.unsqueeze(1)                       
                    
                    context = self.attention(encoder_output, decoder_output)
                    
                    decoder_output, decoder_hidden, word_softmax_projection = self.decoder.forward_step(decoder_input, decoder_hidden, context)                    
                    word_softmax_projection = word_softmax_projection.squeeze(1) # eliminate dim 1
                    if print_example:                        
                        _, mi = word_softmax_projection[0].max(0)
                        example_array.append(mi.item())
                        
                    target_y = y[:,i] # select from y the ith column and shape as an array                    
                    loss += self.criterion(word_softmax_projection, target_y) 
                    
                print("\t\t\t Eval Loss: {}".format(loss.data.item()))
                if print_example:
                    print_example = False                        
                    print("----- X:")
                    print(" ".join([self.i2w[str(wi.data.item())] for wi in x[0]]))                        
                    print("----- Y:")
                    print(" ".join([self.i2w[str(wi.data.item())] for wi in y[0]]))
                    print("----- US:")
                    print(" ".join([self.i2w[str(wi)] for wi in example_array]))
                    self.writer.add_text('EvalText', " ".join([self.i2w[str(wi.data.item())] for wi in y[0]]) + " --vs-- "+" ".join([self.i2w[str(wi)] for wi in example_array]))
                    
        elapsed_time = time.time() - start_time    
        print("\t Elapsed time: {}".format(elapsed_time))
            
    def load_checkpoint(self, filename, latest = True):
        if latest: # filename is a folder            
            import glob            
            files = glob.glob(os.path.join(filename,"*.ckp"))
            if files == None:
                raise Exception("Load checkpoint failed with latest=True. Returned list of files in folder [{}] is None".format(filename))            
            filename = sorted(files)[-1]            
            print("Loading latest model {} ...".format(filename))
        else:
            print("Loading model {} ...".format(filename))
        
        checkpoint = torch.load(filename)        
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])        
        self.attention.load_state_dict(checkpoint["attention_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.w2i = checkpoint["w2i"]
        self.w2i = checkpoint["w2i"]
        self.teacher_forcing_ratio = checkpoint["teacher_forcing_ratio"]
        self.epoch = checkpoint["epoch"]
        self.gradient_clip = checkpoint["gradient_clip"]
        epoch = int(os.path.basename(filename).replace("model.","").replace(".ckp",""))        
        return epoch

    def save_checkpoint(self, folder_path, epoch):
        #print("epoch = %d, loss = %f, time = %f" % (epoch, loss, time))
        filename = os.path.join(folder_path,"model."+str(epoch).rjust(4,"0")+".ckp")
        checkpoint = {}
        checkpoint["encoder_state_dict"] = self.encoder.state_dict()
        checkpoint["decoder_state_dict"] = self.decoder.state_dict()
        checkpoint["attention_state_dict"] = self.attention.state_dict()
        checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        checkpoint["w2i"] = self.w2i
        checkpoint["i2w"] = self.i2w
        checkpoint["teacher_forcing_ratio"] = self.teacher_forcing_ratio
        checkpoint["epoch"] = self.epoch
        checkpoint["gradient_clip"] = self.gradient_clip
        torch.save(checkpoint, filename)       

"""
x = torch.rand(3,4)
y = torch.rand(3,4)
print(x)
_, argmax = x.max(1)
argmax = argmax.unsqueeze(1)
print(argmax.size())
"""

""" input, target
tensor([[ 0.2375, -0.1487, -0.0832, -0.7523, -0.5504],
        [ 0.6835, -0.4430,  0.3776, -0.4433, -0.2840],
        [-0.3039,  2.1489, -1.7675, -0.9361, -0.3278]], requires_grad=True)
tensor([4, 0, 3])
criterion = nn.CrossEntropyLoss()
criterion(input, target)
"""
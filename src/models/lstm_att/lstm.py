from pprint import pprint
import time, io, os, random

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import SimpleLSTMEncoderLayer, SimpleLSTMDecoderLayer, AttentionLayer
from progressbar import ProgressBar

#from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class LSTMEncoderDecoderAtt(nn.Module):
    def __init__(self, w2i, i2w, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, encoder_n_layers, decoder_n_layers, encoder_drop_prob=0.5, decoder_drop_prob=0.5, lr = 0.01, teacher_forcing_ratio=0.5, gradient_clip = 5, model_store_path = None):
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
        self.lr = lr
        self.vocab_size = len(w2i)
        print("Vocab size: {}".format(self.vocab_size))
        
        self.train_on_gpu=torch.cuda.is_available()        
        if(self.train_on_gpu):
            print('Training on GPU.')
        else:
            print('No GPU available, training on CPU.')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_store_path == None:
            self.model_store_path = os.path.dirname(os.path.realpath(__file__))
        else:
            self.model_store_path = model_store_path
        if not os.path.exists(model_store_path):
            os.makedirs(model_store_path)
        #self.writer = SummaryWriter('/work/tmp')        
        
    def show_tensor(x, prediction=None, source=None): # x is a numpy 2d matrix
        fig = plt.figure(figsize=(12, 6))
        sns.heatmap(x,cmap="rainbow")
        plt.tight_layout()        
        return fig            
            
    def train(self, train_loader, valid_loader, test_loader, batch_size, patience = 10):                           
        current_patience = patience
        
        # move model to GPU, if available
        if(self.train_on_gpu):
            self.encoder.cuda()
            self.decoder.cuda()
            self.attention.cuda()
        
        best_loss = 1000000.
        best_epoch = -1
        while current_patience > 0:                  
            current_patience -= 1
            train_loss = self._train_epoch(train_loader, batch_size)            
            self.save_checkpoint("last")
            
            eval_loss = self._eval(valid_loader, batch_size)
            if eval_loss < best_loss:
                current_patience = patience
                best_loss = eval_loss
                best_epoch = self.epoch
                self.save_checkpoint("best")
            
            print("\nEpoch {} training loss {}, eval loss {}, best loss {} at epoch {}\n".format(self.epoch, train_loss, eval_loss, best_loss, best_epoch))
            
            
    def _train_epoch(self, train_loader, batch_size):                       
        self.epoch += 1
        self.encoder.train()
        self.decoder.train()
        self.attention.train()        
        
        encoder_hidden = self.encoder.init_hidden(batch_size)
        decoder_hidden = self.decoder.init_hidden(batch_size)
        total_loss = 0.
        pbar = ProgressBar()
        pbar.set(total_steps=len(train_loader)) 
        
        for counter, (x, y) in enumerate(train_loader):
            max_seq_len_x = x.size(1) # x este 64 x 399 (variabil)
            max_seq_len_y = y.size(1) # y este 64 x variabil
            
            pbar.update(progress=counter, text="Epoch {:d}, progress {}/{}, train average loss \033[93m{:.6f}\033[0m (mx/my = {}/{}) ... ".format(self.epoch, counter, len(train_loader), total_loss/(counter+1), max_seq_len_x, max_seq_len_y))                         
                        
            #if counter > 1:               
            #    break                
            if counter % 1000 == 0 and counter > 0:
                self.save_checkpoint("last")
            
            
            loss = 0
            #print("  Epoch {}, batch: {}/{}, max_seq_len_x: {}, max_seq_len_y: {}".format(self.epoch, counter, len(train_loader), max_seq_len_x, max_seq_len_y))
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
                    decoder_input = torch.zeros(batch_size, 1, dtype = torch.long, device=self.device) # 1 in middle is because lstm expects (batch, seq_len, input_size): 
                    for j in range(batch_size):
                        decoder_input[j]=y[j][i]                
                    #print(decoder_input.size()) # batch_size x 1                            
                else: # feed own previous prediction extracted from word_softmax_projection
                    _, decoder_input = word_softmax_projection.max(1) # no need for values, just indexes 
                    decoder_input = decoder_input.unsqueeze(1) # from batch_size to batch_size x 1                    
                    #print(decoder_input.size()) # batch_size x 1                            

                # remove me, for printing attention
                if counter == 1:
                    self.attention.should_print = False#True
                    #print("\t Decoder step {}/{}".format(i, max_seq_len_y))    
                else:
                    self.attention.should_print = False
                    self.attention.att_mat = []
                context = self.attention(encoder_output, decoder_output)
                
                # context is batch_size * encoder_hidden_dim            
                decoder_output, decoder_hidden, word_softmax_projection = self.decoder.forward_step(decoder_input, decoder_hidden, context)
                # first, reduce word_softmax_projection which is torch.Size([64, 1, 50004]) to 64 * 50004
                word_softmax_projection = word_softmax_projection.squeeze(1) # eliminate dim 1
                
                # now, select target y
                # y looks like batch_size * max_seq_len_y : tensor([[    2, 10890, 48108,  ...,     0,     0,     0], ... ... ..
                target_y = y[:,i] # select from y the ith column and shape as an array 
                # target_y now looks like [ 10, 2323, 5739, 24, 9785 ... ] of size 64 (batch_size)
                #print(word_softmax_projection.size())
                #print(target_y.size())
                loss += self.criterion(word_softmax_projection, target_y) # ignore index not set as we want 0 to count to error too
            
            # remove me, attention printing
            """if counter == 1:
                fig = plt.figure(figsize=(12, 10))
                sns.heatmap(self.attention.att_mat,cmap="gist_heat")                
                plt.tight_layout()            
                fig.savefig('img/__'+str(self.epoch)+'.png')
                plt.clf()
            """    
            total_loss += loss.data.item()
            loss.backward() # calculate the loss and perform backprop
            
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.gradient_clip)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), self.gradient_clip)
            nn.utils.clip_grad_norm_(self.attention.parameters(), self.gradient_clip)
            self.optimizer.step()
            
            #self.writer.add_scalar('Train/Loss', loss.data.item())            
            #break
        
        pbar.update(text="Epoch {:d}, train done, average loss \033[93m{:.6f}\033[0m".format(self.epoch, total_loss/len(train_loader))) 

        return total_loss/len(train_loader)
    
    def run (self, data_loader, batch_size, beam_size=3): #data is either a list of lists or a dataset_loader
        self.encoder.eval()
        self.decoder.eval()
        self.attention.eval()            
        
        pbar = ProgressBar()
        pbar.set(total_steps=len(data_loader)) 
       
        total_loss = 0.
        with torch.no_grad():
            for counter, (x, y) in enumerate(data_loader):                
                pbar.update(progress=counter, text="Epoch {:d}, progress {}/{}, eval average loss \033[93m{:.6f}\033[0m ... ".format(self.epoch, counter, len(data_loader), total_loss/(counter+1)))  
                
                if x.size(0) != batch_size:
                    print("\t Incomplete batch, skipping.")
                    continue
                
                if(self.train_on_gpu):
                    x, y = x.cuda(), y.cuda()
                
                x = x[0:1,:]                
                y = y[0:1,:]
                results, scores, loss = self._run_instance(x, y, beam_size)
        
        pbar.update(text="Epoch {:d}, eval done, average loss \033[93m{:.6f}\033[0m".format(self.epoch, total_loss/len(data_loader)))     
        return total_loss/len(data_loader)
        
    def _run_instance (self, x, y, beam_size):        
        from layers import Beam
        max_seq_len_x = x.size(1)
        max_seq_len_y = y.size(1)
        loss = 0
        
        # encoder
        encoder_hidden = self.encoder.init_hidden(batch_size=1)
        encoder_output, encoder_hidden = self.encoder(x, encoder_hidden) 
        
        # decoder hidden init
        (d_hid, d_cell) = self.decoder.init_hidden(batch_size=beam_size)
        # split into hidden and cell states, and format into #torch.Size([2, 1, 64, 512])
        #d_a = decoder_hidden[0].view(self.decoder_n_layers, 1, beam_size, self.decoder_hidden_dim)
        #d_b = decoder_hidden[1].view(self.decoder_n_layers, 1, beam_size, self.decoder_hidden_dim)
                
        # init decoders (beam_size)
        beams = []
        for i in range(beam_size):        
            b = Beam()            
            #print( d_hid.size() ) # torch.Size([1, 3, 256]) 1 layer, 3 batch_size, 256 hidden
            b.current_decoder_hidden = (d_hid[:,i:i+1,:], d_cell[:,i:i+1,:])            
            b.sequence = [3] # set to BOS, which is 2, 3 is for dummy loader            
            beams.append(b)
            if i != 0: # force that in the first step all results come from the first beam
                b.score = -10000
            
                        
        #word_softmax_projection = torch.zeros(1, 5, dtype = torch.float, device=self.device)
        #word_softmax_projection[:,2] = 1. # beginning of sentence value is 2, set it  #XXX
        
        # prepare decoder for initial attention computation
        #decoder_output = decoder_hidden[0].view(self.decoder_n_layers, 1, beam_size, self.decoder_hidden_dim) #torch.Size([2, 1, 64, 512])
        decoder_output = d_hid.view(self.decoder_n_layers, 1, beam_size, self.decoder_hidden_dim) #torch.Size([2, 1, 64, 512])
        decoder_output = decoder_output[-1].permute(1,0,2) 
                        
        loss = 0 
        total_loss = 0                
        example_array = []
        
        for i in range(max_seq_len_y): 
            print("\n\n\t Decoder step {}/{}".format(i, max_seq_len_y))                        
            
            # for decoder we need decoder_input, decoder_hidden and context
            # start with decoder_input: it is a batch_size * 1 containing 1 word index (previous)
            decoder_input_list = []
            for j in range(beam_size):        
                decoder_input_list.append([beams[j].sequence[-1]]) # select last word for each beam            
            decoder_input = torch.LongTensor(decoder_input_list, device = self.device)
            
            # compose decoder_hidden
            # final hidden should be tuple of ( torch.Size([1, 3, 256]), torch.Size([1, 3, 256]) ), meaning layers, beam_size, hidden_size
            d_hid, d_cell = beams[0].current_decoder_hidden[0], beams[0].current_decoder_hidden[1]
            #print(d_hid.size()) # this should be [1, 1, 256]            
            for j in range(1, beam_size): # now, vertically stack others so we get to [1, beam_size, 256] incrementally
                d_hid = torch.cat((d_hid, beams[j].current_decoder_hidden[0]),dim = 1)
                d_cell = torch.cat((d_cell, beams[j].current_decoder_hidden[1]),dim = 1)
                #print(d_hid.size())
            decoder_hidden = (d_hid, d_cell)
            # calculate context for each
            context = self.attention(encoder_output, decoder_output)
            
            #_, decoder_input = word_softmax_projection.max(1) # no need for values, just indexes 
            #decoder_input = decoder_input.unsqueeze(1)                                           
                        
            decoder_output, decoder_hidden, word_softmax_projection = self.decoder.forward_step(decoder_input, decoder_hidden, context)                                
            word_softmax_projection = word_softmax_projection.squeeze(1) # eliminate dim 1            
            #print(word_softmax_projection.size()) # size beam_size x vocab_size
            
            # check for stopping condition                        
            stopped_count = 0
            beam_scores = []
            for j in range(beam_size):
                _, mi = word_softmax_projection[j].max(0)
                if mi == 0: # PAD token, meaning this beam has finished
                    stopped_count +=1 
                    beam_scores.append([-10000]) # ensure no score gets selected from this beam
                else: 
                    beam_scores.append([beams[j].normalized_score()])
            if stopped_count == beam_size:
                print("Reached all beams predicted zero - early condition.")
                break
            
            #print(word_softmax_projection)    
            word_softmax_projection = F.softmax(word_softmax_projection, dim = 1)                
            word_softmax_projection = torch.log(word_softmax_projection) # logarithm of softmax scores
            beam_scores = torch.FloatTensor(beam_scores, device = self.device) # size beam_size x 1
            word_softmax_projection = word_softmax_projection + beam_scores # add logarithms            
                        
            # now, select top scoring values
            flattened_projection = word_softmax_projection.view(beam_size*self.vocab_size)
            max_scores, max_indices = torch.topk(flattened_projection, k = beam_size)
            max_scores = max_scores.cpu().numpy()
            max_indices = max_indices.cpu().numpy()
            
            # identify to which beam each one belongs to, and recreate beams
            new_beams = []
            for (score, index) in zip(max_scores, max_indices):
                belongs_to_beam = int(index/self.vocab_size)
                vocab_index = index%self.vocab_size
                print("Score {}, index {}, belongs to beam {}, vocab_index {}".format(score, index, belongs_to_beam, vocab_index))
                b = Beam()
                b.current_decoder_hidden = (decoder_hidden[0][:,belongs_to_beam:belongs_to_beam+1,:], decoder_hidden[1][:,belongs_to_beam:belongs_to_beam+1,:])
                b.sequence = beams[belongs_to_beam].sequence + [vocab_index]
                b.score = score
                new_beams.append(b)            
            beams = new_beams
            
            print(y.cpu().numpy()[0])
            for b in beams:
                print(str(b.sequence) + " " + str(b.normalized_score()))
            
            #if print_example:                        
            #    _, mi = word_softmax_projection[0].max(0)
            #    example_array.append(mi.item())
                
            #target_y = y[:,i] # select from y the ith column and shape as an array                    
            #loss += self.criterion(word_softmax_projection, target_y) 
        
        #total_loss += loss.data.item()  
        
        sequences = [ b.sequence for b in beams ]
        scores = [ b.normalized_score for b in beams ]
        
        return sequences, scores, total_loss
         
    
    def _run_batch_not_working (self, x, y, beam_size):
        batch_size = x.size(0)
        max_seq_len_x = x.size(1)
        max_seq_len_y = y.size(1)
        loss = 0
        
        encoder_hidden = self.encoder.init_hidden(batch_size)
        decoder_hidden = self.decoder.init_hidden(batch_size)
        
        encoder_output, encoder_hidden = self.encoder(x, encoder_hidden) 
        word_softmax_projection = torch.zeros(batch_size, 5, dtype = torch.float, device=self.device)
        word_softmax_projection[:,2] = 1. # beginning of sentence value is 2, set it  #XXX
        
        decoder_output = decoder_hidden[0].view(self.decoder_n_layers, 1, batch_size, self.decoder_hidden_dim) #torch.Size([2, 1, 64, 512])
        decoder_output = decoder_output[-1].permute(1,0,2) 
                        
        loss = 0 
        total_loss = 0        
        print_example = True
        example_array = []
        
        for i in range(max_seq_len_y): 
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
        
        total_loss += loss.data.item()  
        return [], total_loss
        
    def _eval(self, valid_loader, batch_size):                
        self.encoder.eval()
        self.decoder.eval()
        self.attention.eval()            
        encoder_hidden = self.encoder.init_hidden(batch_size)
        decoder_hidden = self.decoder.init_hidden(batch_size)
        
        pbar = ProgressBar()
        pbar.set(total_steps=len(valid_loader)) 
       
        counter = 0 
        total_loss = 0.
        with torch.no_grad():
            for counter, (x, y) in enumerate(valid_loader):                
                #if counter > 5:
                #    break
                pbar.update(progress=counter, text="Epoch {:d}, progress {}/{}, eval average loss \033[93m{:.6f}\033[0m ... ".format(self.epoch, counter, len(valid_loader), total_loss/(counter+1)))   
                
                max_seq_len_x = x.size(1)
                max_seq_len_y = y.size(1)
                loss = 0
                #print("  Epoch {}, batch: {}/{}, max_seq_len_x: {}, max_seq_len_y: {}".format(self.epoch, counter, len(valid_loader), max_seq_len_x, max_seq_len_y))
                if x.size(0) != batch_size:
                    print("\t Incomplete batch, skipping.")
                    continue
                
                if(self.train_on_gpu):
                    x, y = x.cuda(), y.cuda()
                
                encoder_hidden = tuple([each.data for each in encoder_hidden])
                decoder_hidden = tuple([each.data for each in decoder_hidden])

                encoder_output, encoder_hidden = self.encoder(x, encoder_hidden) 
                word_softmax_projection = torch.zeros(batch_size, 5, dtype = torch.float, device=self.device)
                word_softmax_projection[:,2] = 1. # beginning of sentence value is 2, set it  #XXX
                
                decoder_output = decoder_hidden[0].view(self.decoder_n_layers, 1, batch_size, self.decoder_hidden_dim) #torch.Size([2, 1, 64, 512])
                decoder_output = decoder_output[-1].permute(1,0,2) 
                                
                loss = 0             
                print_example = True
                example_array = []
                
                for i in range(max_seq_len_y): 
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
                
                total_loss += loss.data.item()    
                
                #print("\t\t\t Eval Loss: {}".format(loss.data.item()))
                if print_example:
                    print_example = False 
                    print()                    
                    print("\n\n----- X:")
                    print(" ".join([self.i2w[str(wi.data.item())] for wi in x[0]]))                                            
                    print("----- Y:")
                    print(" ".join([self.i2w[str(wi.data.item())] for wi in y[0]]))                    
                    print("----- OUR PREDICTION:")
                    print(" ".join([self.i2w[str(wi)] for wi in example_array]))
                    print()
                    print(" ".join([str(wi.data.item()) for wi in y[0]]))
                    print(" ".join([str(wi) for wi in example_array]))
                    print()
                    #self.writer.add_text('EvalText', " ".join([self.i2w[str(wi.data.item())] for wi in y[0]]) + " --vs-- "+" ".join([self.i2w[str(wi)] for wi in example_array]), self.epoch)                    
        
        pbar.update(text="Epoch {:d}, eval done, average loss \033[93m{:.6f}\033[0m".format(self.epoch, total_loss/len(valid_loader))) 
    
        return total_loss/len(valid_loader)
    
    def old_run (self, input, max_output_len = 1000): # input is a list of lists of integers (variable len)
        self.encoder.eval()
        self.decoder.eval()
        self.attention.eval()          
        
        batch_size = len(input)
        encoder_hidden = self.encoder.init_hidden(batch_size)
        decoder_hidden = self.decoder.init_hidden(batch_size)       
            
        bordered_input = [ [2]+inst+[3] for inst in input ] # put start and end of sentence markers for each instance
        max_len = max(len(inst) for inst in bordered_input) # determines max size for all examples 

        input = np.array( [ inst + [0] * (max_len - len(inst)) for inst in bordered_input ] ) # input is now a max_len object padded with zeroes to the right (for all instances)        
        
        with torch.no_grad():            
            # numpy to tensor            
            x = torch.LongTensor(input)
            if(self.train_on_gpu):
                x = x.cuda()
            
                
            encoder_output, encoder_hidden = self.encoder(x, encoder_hidden) 
            word_softmax_projection = torch.zeros(batch_size, 5, dtype = torch.float, device=self.device)
            word_softmax_projection[:,2] = 1. # beginning of sentence value is 2, set it #XXX remember to put 2 instead of 3 for non-dummy 
            
            decoder_output = decoder_hidden[0].view(self.decoder_n_layers, 1, batch_size, self.decoder_hidden_dim) 
            decoder_output = decoder_output[-1].permute(1,0,2) 
                            
            output = [ [] for _ in range(batch_size) ]            
            for i in range(max_output_len):                
                _, decoder_input = word_softmax_projection.max(1) # no need for values, just indexes 
                decoder_input = decoder_input.unsqueeze(1)                                           
                context = self.attention(encoder_output, decoder_output)
                
                decoder_output, decoder_hidden, word_softmax_projection = self.decoder.forward_step(decoder_input, decoder_hidden, context)                    
                word_softmax_projection = word_softmax_projection.squeeze(1) # eliminate dim 1
                
                zero_count = 0
                for j in range(batch_size):
                    _, mi = word_softmax_projection[j].max(0)                    
                    output[j].append(mi.cpu().item())
                    if mi.item() == 0:
                        zero_count += 1
                
                # check ending condition (all zeroes)
                if zero_count == batch_size:
                    break
        
        # transform back to numpy (and move back to CPU just in case it was on GPU)
        #output = output.numpy()
        
        # clean each array
        clean_output = []
        for instance in output:
            clean_instance = []
            for element in instance:
                if element > 3:
                    clean_instance.append(element)
            clean_output.append(clean_instance)
        
        return clean_output
        
    
    def load_checkpoint(self, filename):
        """if latest: # filename is a folder            
            import glob            
            files = glob.glob(os.path.join(filename,"*.ckp"))
            if files == None:
                raise Exception("Load checkpoint failed with latest=True. Returned list of files in folder [{}] is None".format(filename))            
            filename = sorted(files)[-1]            
            print("Loading latest model {} ...".format(filename))                   
        """
        filename = os.path.join(self.model_store_path,"model."+filename+".ckp")
        print("Loading model {} ...".format(filename))
        checkpoint = torch.load(filename)        
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])        
        self.attention.load_state_dict(checkpoint["attention_state_dict"])
        #self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.w2i = checkpoint["w2i"]
        self.i2w = checkpoint["i2w"]
        self.teacher_forcing_ratio = checkpoint["teacher_forcing_ratio"]
        self.epoch = checkpoint["epoch"]
        self.gradient_clip = checkpoint["gradient_clip"]        

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.attention.to(self.device)
        #self.optimizer.to(self.device) # careful to continue training on the same device !
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters())+list(self.attention.parameters()), lr=self.lr)        
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        
        
    def save_checkpoint(self, filename):        
        filename = os.path.join(self.model_store_path,"model."+filename+".ckp")
        
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
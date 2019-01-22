import torch
import torch.nn as nn

from layers import SimpleLSTMEncoderLayer, SimpleLSTMDecoderLayer
    
class LSTMEncoderDecoder(nn.Module):
    def __init__(self, w2i, i2w, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, encoder_n_layers, decoder_n_layers, encoder_drop_prob=0.5, decoder_drop_prob=0.5, lr = 0.01):
        super(LSTMEncoderDecoder, self).__init__()
        
        self.encoder = SimpleLSTMEncoderLayer(len(w2i), embedding_dim, encoder_hidden_dim, encoder_n_layers, encoder_drop_prob)
        self.decoder = SimpleLSTMDecoderLayer(encoder_hidden_dim, decoder_hidden_dim, len(w2i), decoder_n_layers, decoder_drop_prob)
        
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        
        self.w2i = w2i
        self.i2w = i2w
        
        self.train_on_gpu=torch.cuda.is_available()
        if(self.train_on_gpu):
            print('Training on GPU.')
        else:
            print('No GPU available, training on CPU.')
        
    def train(self, train_loader, valid_loader, test_loader, batch_size):        
        print("\nStart training ...")
        
        #criterion = nn.BCELoss()
        
        # training params
        epochs = 40 # 3-4 is approx where I noticed the validation loss stop decreasing

        counter = 0
        print_every = 100
        clip=5 # gradient clipping
                
        # move model to GPU, if available
        if(self.train_on_gpu):
            self.encoder.cuda()
            self.decoder.cuda()

        self.encoder.train()
        self.decoder.train()
        
        # train for some number of epochs
        for e in range(epochs):                        
            encoder_hidden = self.encoder.init_hidden(batch_size)
            decoder_hidden = self.decoder.init_hidden(batch_size)

            # batch loop
            for x, y in train_loader:            
                #print(x)
                counter += 1

                if(self.train_on_gpu):
                    x, y = inputs.x(), y.cuda()

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                encoder_hidden = tuple([each.data for each in encoder_hidden])
                decoder_hidden = tuple([each.data for each in decoder_hidden])

                # zero grads in optimizer
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                
                # forward
                encoder_output, encoder_hidder = self.encoder(x, encoder_hidden)
                decoder_output, decoder_hidder = self.decoder(encoder_output, decoder_hidden)
                
                # detach??
                
                # calculate the loss and perform backprop
                loss = criterion(output.squeeze(), y.float())
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(net.parameters(), clip)
                optimizer.step()

                # loss stats
                if counter % print_every == 0:
                    # Get validation loss
                    val_h = net.init_hidden(batch_size)
                    val_losses = []
                    net.eval()
                    for inputs, labels in valid_loader:

                        # Creating new variables for the hidden state, otherwise
                        # we'd backprop through the entire training history
                        val_h = tuple([each.data for each in val_h])

                        if(train_on_gpu):
                            inputs, labels = inputs.cuda(), labels.cuda()

                        output, val_h = net(inputs, val_h)
                        val_loss = criterion(output.squeeze(), labels.float())

                        val_losses.append(val_loss.item())

                    net.train()
                    print("Epoch: {}/{}...".format(e+1, epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))
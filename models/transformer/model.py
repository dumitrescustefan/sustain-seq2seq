import os
import torch.nn as nn
import torch
import numpy as np
from math import sin, cos
from components import Attention


class Transformer(nn.Module):
    def __init__(self, N, d_model, d_ff, h, d_k, d_v, n_class, voc_size):
        super(Transformer, self).__init__()

        self.N = N

        if torch.cuda.is_available():
            print('Running on GPU.')
            self.cuda = True
            self.device = torch.device('cuda')            
        else:
            print('Running on CPU.')
            self.cuda = False
            self.device = torch.device('cpu')        
        
        self.N = N
        self.pos_embeddings = self._generate_pos_embeddings(voc_size, d_model)        
        self.embedding_layer = nn.Embedding(n_class, d_model)

        self.encoder_units = [Encoder(d_model, d_ff, h, d_k, d_v, self.device) for _ in range(N)]
        self.decoder_units = [Decoder(d_model, d_ff, h, d_k, d_v, self.device) for _ in range(N)]
        
        self.linear = nn.Linear(d_model, n_class)
        if self.cuda:
            self.to(self.device)
        
    def _generate_pos_embeddings(self, seq_len, embeddings_size):
        pos_embeddings = np.zeros((seq_len, embeddings_size))#torch.Tensor(size=[seq_len, embeddings_size])

        for pos in range(seq_len):
            for i in range(embeddings_size):
                if i % 2 == 0:
                    pos_embeddings[pos][int(i/2)] = sin(pos/10000**(2*i/embeddings_size))
                else:
                    pos_embeddings[pos][int((embeddings_size + i)/2)] = cos(pos/10000**(2*(i-1)/embeddings_size))
        
        if self.cuda:
            return torch.Tensor(pos_embeddings).cuda()
        else:
            return torch.Tensor(pos_embeddings)

    def forward(self, x, y):
        x_seq_len = x.shape[1]
        y_seq_len = y.shape[1]

        encoder_data = torch.add(self.embedding_layer(x), self.pos_embeddings[0:x_seq_len])
        decoder_data = torch.add(self.embedding_layer(y), self.pos_embeddings[0:y_seq_len])

        for i in range(self.N):
            encoder_data = self.encoder_units[i].forward(encoder_data)

        for i in range(self.N):
            decoder_data = self.decoder_units[i].forward(decoder_data, encoder_data)

        output = self.linear(decoder_data)

        return output

    def eval(self):
        super().eval()

        for i in range(self.N):
            self.encoder_units[i].eval()
            self.decoder_units[i].eval()

    def train(self, mode=True):
        super().train(mode)

        for i in range(self.N):
            self.encoder_units[i].train(mode)
            self.decoder_units[i].train(mode)

    def to(self, device):
        super().to(device)

        for i in range(self.N):
            self.encoder_units[i].to(device)
            self.decoder_units[i].to(device)

    def load_checkpoint(self, folder, extension):
        filename = os.path.join(folder, "checkpoint." + extension)
        print("Loading model {} ...".format(filename))
        if not os.path.exists(filename):
            print("\t Model file not found, not loading anything!")
            return {}

        checkpoint = torch.load(filename)        
        self.load_state_dict(checkpoint["self_state_dict"])
        for i in range(self.N):
            self.encoder_units[i].load_state_dict(checkpoint["encoder"][i])
            self.decoder_units[i].load_state_dict(checkpoint["decoder"][i])        
        self.to(self.device)
        return checkpoint["extra"]

    def save_checkpoint(self, folder, extension, extra={}):
        filename = os.path.join(folder, "checkpoint." + extension)
        checkpoint = {}        
        checkpoint["self_state_dict"] = self.state_dict()
        checkpoint["encoder"] = []
        checkpoint["decoder"] = []
        for i in range(self.N):
            checkpoint["encoder"].append(self.encoder_units[i].state_dict())
            checkpoint["decoder"].append(self.decoder_units[i].state_dict())            
        checkpoint["extra"] = extra
        torch.save(checkpoint, filename)


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, h, d_k, d_v, device):
        super(Encoder, self).__init__()
        self.selfattention_layer = Attention(d_model, h, d_k, d_v, device)
        self.batch_norm1 = nn.BatchNorm1d(d_model)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.batch_norm2 = nn.BatchNorm1d(d_model)

        self.to(device)

    def forward(self, input):
        selfattn_output = self.selfattention_layer.forward(input, input, input)
        selfattn_output = torch.add(selfattn_output, input)
        selfattn_output = self.batch_norm1(selfattn_output.permute(0, 2, 1)).permute(0, 2, 1)

        hidden = self.linear1(selfattn_output)
        output = self.linear2(hidden)
        output = torch.add(output, selfattn_output)
        output = self.batch_norm2(output.permute(0, 2, 1)).permute(0, 2, 1)

        return output


class Decoder(nn.Module):
    def __init__(self,  d_model, d_ff, h, d_k, d_v, device):
        super(Decoder, self).__init__()
        self.selfattention_layer = Attention(d_model, h, d_k, d_v, device)
        self.batch_norm1 = nn.BatchNorm1d(d_model)

        self.encoderdecoderattention = Attention(d_model, h, d_k, d_v, device)
        self.batch_norm2 = nn.BatchNorm1d(d_model)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.batch_norm3 = nn.BatchNorm1d(d_model)
        
        self.to(device)
        
    def forward(self, input, encoder_data):
        selfattn_output = self.selfattention_layer.forward(input, input, input)
        selfattn_output = torch.add(selfattn_output, input)
        selfattn_output = self.batch_norm1(selfattn_output.permute(0, 2, 1)).permute(0, 2, 1)

        encdecattn_output = self.encoderdecoderattention.forward(selfattn_output, encoder_data, encoder_data)
        encdecattn_output = torch.add(encdecattn_output, selfattn_output)
        encdecattn_output = self.batch_norm2(encdecattn_output.permute(0, 2, 1)).permute(0, 2, 1)

        hidden = self.linear1(encdecattn_output)
        output = self.linear2(hidden)
        output = torch.add(output, encdecattn_output)
        output = self.batch_norm2(output.permute(0, 2, 1)).permute(0, 2, 1)

        return output